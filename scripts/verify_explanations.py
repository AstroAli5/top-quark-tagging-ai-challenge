"""Independently verify MATLAB explanation tables against saved probabilities."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import rankdata


def auc(probabilities, labels):
    probabilities = np.asarray(probabilities).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if not np.isfinite(probabilities).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid predictions or labels')
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Probabilities must be in [0, 1]')
    positive = labels == 1
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        raise ValueError('Both classes are required')
    return float((rankdata(probabilities)[positive].sum() - n1*(n1+1)/2)/(n1*n0))


def read_table(path, fields):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    result = {tuple(row[field] for field in fields): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate metric rows in {path}')
    return result


def close(actual, expected):
    if not np.isclose(actual, float(expected), rtol=0, atol=1e-12):
        raise ValueError(f'Metric mismatch: {actual} != {expected}')


def verify(directory, core_input):
    directory, core_input = Path(directory), Path(core_input)
    meta = json.loads((directory/'metadata.json').read_text())
    seeds = np.atleast_1d(meta['trainingSeeds']).astype(int).tolist()
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate training seeds')
    provenance = meta['provenance']
    if isinstance(provenance, dict):
        provenance = [provenance]
    n = meta['testJets']
    permutation = read_table(directory/'permutation_per_repeat.csv', ['Seed','PermutationSeed','Feature'])
    occlusion = read_table(directory/'occlusion_per_seed.csv', ['Seed','RadiusFractionKept'])
    feature_summary = read_table(directory/'feature_summary.csv', ['Feature'])
    radius_summary = read_table(directory/'occlusion_summary.csv', ['RadiusFractionKept'])
    feature_values, radius_values, checked_permutation, checked_occlusion = {}, {}, set(), set()
    for seed, source in zip(seeds, provenance, strict=True):
        raw = loadmat(directory/f'predictions_seed_{seed}.mat', simplify_cells=True)
        core = core_input/f'research-core-seed-{seed}'
        original = loadmat(core/'results/clean_predictions.mat', simplify_cells=True)
        # MATLAB string objects are opaque to SciPy; use the saved JSON schema.
        names = json.loads((core/'results/metadata.json').read_text())['models']
        columns = [names.index('CNN'), names.index('GraphSAGE')]
        labels, rows = np.asarray(raw['labels']), np.asarray(raw['rows'])
        np.testing.assert_array_equal(rows, np.arange(n))
        np.testing.assert_array_equal(rows, original['rows'][:n])
        np.testing.assert_array_equal(labels, original['labels'][:n])
        clean = np.asarray(raw['cleanProbabilities'])
        tolerance = 64*float(np.finfo(np.float32).eps)
        close(tolerance,meta['cleanProbabilityTolerance'])
        np.testing.assert_allclose(clean, original['probabilities'][:n, columns], rtol=0, atol=tolerance)
        np.testing.assert_array_equal(clean>=0.5,original['probabilities'][:n, columns]>=0.5)
        close(np.max(np.abs(clean-original['probabilities'][:n, columns])), source['maxCleanProbabilityDifference'])
        for filename, key in [('cnn_model.mat','cnnCheckpointSHA256'), ('graphsage_model.mat','graphCheckpointSHA256')]:
            with (core/'models'/filename).open('rb') as handle:
                digest = hashlib.file_digest(handle,'sha256').hexdigest()
            if digest != source[key]:
                raise ValueError(f'Checkpoint hash mismatch: seed {seed}, {filename}')
        graph_base, cnn_base = auc(clean[:,1], labels), auc(clean[:,0], labels)
        for m, baseline in enumerate([cnn_base,graph_base]):
            delta = abs(baseline-auc(original['probabilities'][:n,columns[m]],labels))
            close(delta,source['cleanAUCDifference'][m])
            if delta > 1e-6:
                raise ValueError('Clean AUC drift exceeds tolerance')
        features = ['deltaEta','deltaPhi','log(pT)','log(E)']
        repeats = np.atleast_1d(raw['permutationSeeds']).astype(int).tolist()
        radii = np.atleast_1d(raw['radiusFractions']).tolist()
        for f, feature in enumerate(features):
            drops = []
            for r, repeat in enumerate(repeats):
                key = (str(seed), str(repeat), feature)
                recorded = permutation[key]; checked_permutation.add(key)
                score = auc(raw['permutationProbabilities'][:,f,r],labels)
                close(graph_base,recorded['BaselineAUC']); close(score,recorded['PerturbedAUC'])
                close(graph_base-score,recorded['AUCDrop']); close(n,recorded['TestJets'])
                drops.append(graph_base-score)
            feature_values.setdefault(feature,[]).append(float(np.mean(drops)))
        np.testing.assert_array_equal(raw['occlusionProbabilities'][:,0],clean[:,0])
        for r, radius in enumerate(radii):
            key = (str(seed),format(radius,'g'))
            recorded = occlusion[key]; checked_occlusion.add(key)
            score = auc(raw['occlusionProbabilities'][:,r],labels)
            close(cnn_base,recorded['BaselineAUC']); close(score,recorded['AUC'])
            close(cnn_base-score,recorded['AUCDrop']); close(n,recorded['TestJets'])
            radius_values.setdefault(format(radius,'g'),[]).append(score)
    if checked_permutation != set(permutation) or checked_occlusion != set(occlusion):
        raise ValueError('Unexpected or missing perturbation rows')
    for values, table, mean_key in [(feature_values,feature_summary,'MeanAUCDrop'), (radius_values,radius_summary,'MeanAUC')]:
        if set(values) != {key[0] for key in table}:
            raise ValueError('Unexpected summary rows')
        for key, scores in values.items():
            recorded = table[(key,)]
            close(np.mean(scores),recorded[mean_key])
            close(np.std(scores,ddof=1) if len(scores)>1 else 0,recorded['SeedSD'])
            close(len(seeds),recorded['TrainingSeeds'])
    report = {'status':'verified','training_seeds':seeds,'test_jets':n,
              'permutation_metrics_checked':len(checked_permutation),
              'occlusion_metrics_checked':len(checked_occlusion),
              'auc_method':'Independent average-rank calculation, ties receive half credit',
              'checkpoint_hashes_verified':2*len(seeds),'clean_probability_tolerance':tolerance,
              'clean_auc_tolerance':1e-6,'changed_clean_decisions':0,
              'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--core-input',required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.input,args.core_input),indent=2))
