"""Optional, matched-input four-qubit simulator experiment; see its protocol."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.io import loadmat
from scipy.stats import rankdata

FEATURES = ['mass_over_jet_pt', 'girth', 'pt_dispersion', 'log1p_constituents']
MODELS = ['Linear SVM', 'RBF SVM', 'Quantum kernel SVM']
SEEDS = [17, 29, 43]
C_VALUES = [.1, 1., 10., 100.]


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def jet_features(particles):
    p = np.asarray(particles, dtype=float).reshape(-1, 200, 4)
    if not np.isfinite(p).all() or (p[:, :, 0] < 0).any():
        raise ValueError('Nonfinite or negative-energy input')
    pt = np.hypot(p[:, :, 1], p[:, :, 2])
    keep = (p[:, :, 0] > 0) & (pt > 0)
    p = np.where(keep[:, :, None], p, 0.)
    pt = np.where(keep, pt, 0.)
    total = p.sum(axis=1)
    jet_pt = np.hypot(total[:, 1], total[:, 2])
    sum_pt = pt.sum(axis=1)
    if (keep.sum(axis=1) < 3).any() or (jet_pt <= 0).any():
        raise ValueError('Near-empty jet or zero transverse momentum')
    mass = np.sqrt(np.maximum(total[:, 0]**2 - (total[:, 1:]**2).sum(axis=1), 0.))
    eta = np.arcsinh(np.divide(p[:, :, 3], pt, out=np.zeros_like(pt), where=keep))
    jet_eta = np.arcsinh(total[:, 3] / jet_pt)
    phi = np.arctan2(p[:, :, 2], p[:, :, 1])
    jet_phi = np.arctan2(total[:, 2], total[:, 1])
    delta_phi = np.arctan2(np.sin(phi-jet_phi[:, None]), np.cos(phi-jet_phi[:, None]))
    radius = np.hypot(eta-jet_eta[:, None], delta_phi)
    return np.column_stack([mass/jet_pt, (pt*radius).sum(axis=1)/sum_pt,
                            np.sqrt((pt**2).sum(axis=1))/sum_pt, np.log1p(keep.sum(axis=1))])


def balanced_rows(labels, count, seed):
    y = np.asarray(labels).ravel()
    if count < 4 or count % 2 or not np.isin(y, [0, 1]).all():
        raise ValueError('Need an even sample size and binary labels')
    rng = np.random.default_rng(seed)
    rows = np.concatenate([rng.choice(np.flatnonzero(y == cls), count//2, replace=False)
                           for cls in [0, 1]])
    return rows[rng.permutation(len(rows))]


def fidelity_kernel(left, right):
    return np.clip(np.abs(np.asarray(left).conj() @ np.asarray(right).T)**2, 0., 1.)


def encode(values):
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector
    circuit = zz_feature_map(4, reps=2, entanglement='linear')
    states = np.asarray([Statevector.from_instruction(circuit.assign_parameters(row)).data
                         for row in values])
    if not np.allclose(np.linalg.norm(states, axis=1), 1., atol=1e-12, rtol=0):
        raise ValueError('Invalid state normalization')
    return states


def checked_inputs(directory):
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['dataset'] != '10.5281/zenodo.2603256':
        raise ValueError('Unexpected source dataset')
    chunk = manifest['test_chunks'][0]
    if chunk['start'] != 0 or chunk['stop'] != 5000:
        raise ValueError('Pilot requires the original first 5,000-row test chunk')
    hashes = {'fitting.mat': manifest['fitting_sha256'], chunk['file']: chunk['sha256']}
    for name, digest in hashes.items():
        if sha256(root/name) != digest:
            raise ValueError(f'Prepared data checksum mismatch: {name}')
    fit = loadmat(root/'fitting.mat'); test = loadmat(root/chunk['file'])
    parts = fit['partition'].ravel()
    if not np.array_equal(parts, np.r_[np.ones(50000), np.full(10000, 2)]):
        raise ValueError('Pilot requires the recorded 50,000/10,000 fitting pools')
    if not np.array_equal(test['sourceRows'].ravel(), np.arange(5000)):
        raise ValueError('Test rows changed')
    return fit, test, manifest, hashes


def independent_auc(y, scores):
    y = np.asarray(y); scores = np.asarray(scores)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if not n1 or not n0 or not np.isfinite(scores).all():
        raise ValueError('Invalid scores/classes')
    return float((rankdata(scores)[y == 1].sum() - n1*(n1+1)/2)/(n1*n0))


def verify(directory):
    root = Path(directory)
    report = json.loads((root/'report.json').read_text())
    with (root/'test_scores.csv').open() as f:
        rows = list(csv.DictReader(f))
    common = None
    for result in report['per_repeat']:
        selected = [r for r in rows if int(r['sampling_seed']) == result['sampling_seed']]
        identities = [(int(r['source_row']), int(r['label'])) for r in selected]
        if len(identities) != 2000 or len(set(x[0] for x in identities)) != 2000:
            raise ValueError('Incomplete or duplicate test sample')
        if common is None: common = identities
        elif common != identities: raise ValueError('Repeats used different test rows')
        y = np.asarray([int(r['label']) for r in selected])
        score = np.asarray([float(r[result['model']]) for r in selected])
        if not np.isclose(independent_auc(y, score), result['test_auc'], atol=1e-12, rtol=0):
            raise ValueError('AUC does not match saved scores')
        if not np.isclose(np.mean((score > 0) == y), result['test_accuracy'], atol=1e-12, rtol=0):
            raise ValueError('Accuracy does not match saved scores')
    for result in report['summary']:
        for metric in ['test_auc', 'test_accuracy']:
            values = [r[metric] for r in report['per_repeat'] if r['model'] == result['model']]
            if not np.isclose(np.mean(values), result[metric+'_mean'], atol=1e-12, rtol=0):
                raise ValueError('Summary mean changed')
            if not np.isclose(np.std(values, ddof=1), result[metric+'_sd'], atol=1e-12, rtol=0):
                raise ValueError('Summary SD changed')
    checked = {'status': 'verified', 'model_repeat_metrics': len(report['per_repeat']),
               'unique_test_jets': 2000, 'sampling_repeats': 3,
               'method': 'Independent average-rank AUC and threshold-zero accuracy',
               'scores_sha256': sha256(root/'test_scores.csv'), 'verifier_sha256': sha256(__file__)}
    (root/'verification.json').write_text(json.dumps(checked, indent=2)+'\n')
    return checked


def run(directory, output):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score, accuracy_score
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    fit, test, manifest, hashes = checked_inputs(directory)
    labels = fit['labels'].ravel().astype(int)
    pool_train = np.flatnonzero(fit['partition'].ravel() == 1)
    pool_val = np.flatnonzero(fit['partition'].ravel() == 2)
    test_rows = balanced_rows(test['labels'], 2000, 73)
    test_y = test['labels'].ravel()[test_rows].astype(int)
    test_raw = jet_features(test['particleData'][test_rows])
    metrics = []; score_rows = []; provenance = []
    started = time.perf_counter()
    for seed in SEEDS:
        train_rows = pool_train[balanced_rows(labels[pool_train], 512, seed)]
        val_rows = pool_val[balanced_rows(labels[pool_val], 256, seed)]
        train_y = labels[train_rows]; val_y = labels[val_rows]
        train_raw = jet_features(fit['particleData'][train_rows])
        val_raw = jet_features(fit['particleData'][val_rows])
        scaler = MinMaxScaler(feature_range=(0., np.pi), clip=True).fit(train_raw)
        x_train = scaler.transform(train_raw); x_val = scaler.transform(val_raw)
        x_test = scaler.transform(test_raw)
        record = {'sampling_seed': seed, 'train_source_rows': train_rows.tolist(),
                  'validation_source_rows': (val_rows-50000).tolist(),
                  'scaler_min': scaler.data_min_.tolist(), 'scaler_max': scaler.data_max_.tolist(),
                  'test_clipped_feature_fraction': float(np.mean((test_raw < scaler.data_min_) | (test_raw > scaler.data_max_)))}
        scores = {}
        for model in MODELS:
            timer = time.perf_counter()
            if model == 'Quantum kernel SVM':
                train_states = encode(x_train); val_states = encode(x_val)
                k_train = fidelity_kernel(train_states, train_states)
                eigen_min = float(np.linalg.eigvalsh(k_train).min())
                if not np.allclose(k_train, k_train.T, atol=1e-12) or eigen_min < -1e-9 or not np.allclose(np.diag(k_train), 1.):
                    raise ValueError('Invalid training kernel')
                record['minimum_kernel_eigenvalue'] = eigen_min
                train_input = k_train; val_input = fidelity_kernel(val_states, train_states)
                kernel = 'precomputed'
            else:
                train_input, val_input = x_train, x_val
                kernel = 'linear' if model == 'Linear SVM' else 'rbf'
            trials = []
            for c in C_VALUES:
                estimator = SVC(kernel=kernel, C=c, gamma='scale').fit(train_input, train_y)
                auc = float(roc_auc_score(val_y, estimator.decision_function(val_input)))
                trials.append((auc, c, estimator))
            chosen = max(trials, key=lambda item: (item[0], -item[1]))
            # Test features enter model evaluation only after validation selection.
            test_input = fidelity_kernel(encode(x_test), train_states) if kernel == 'precomputed' else x_test
            decisions = chosen[2].decision_function(test_input)
            scores[model] = decisions
            if not np.array_equal(chosen[2].predict(test_input), (decisions > 0).astype(int)):
                raise ValueError('Unexpected SVM class mapping or threshold')
            metrics.append({'sampling_seed': seed, 'model': model, 'selected_c': chosen[1],
                            'validation_auc': chosen[0], 'test_auc': float(roc_auc_score(test_y, decisions)),
                            'test_accuracy': float(accuracy_score(test_y, decisions > 0)),
                            'seconds': time.perf_counter()-timer,
                            'validation_trials': [{'c': c, 'auc': auc} for auc, c, _ in trials]})
        provenance.append(record)
        for i, source_row in enumerate(test_rows):
            score_rows.append({'sampling_seed': seed, 'source_row': int(source_row), 'label': int(test_y[i]),
                               **{name: float(scores[name][i]) for name in MODELS}})
        print(f'Completed sampling repeat {seed}', flush=True)
    summary = []
    for model in MODELS:
        row = {'model': model}
        for metric in ['test_auc', 'test_accuracy', 'seconds']:
            values = [r[metric] for r in metrics if r['model'] == model]
            row.update({metric+'_mean': float(np.mean(values)), metric+'_sd': float(np.std(values, ddof=1))})
        summary.append(row)
    report = {'status': 'completed', 'experiment': 'Four-qubit exact simulator pilot',
              'features': FEATURES, 'train_jets_per_repeat': 512, 'validation_jets_per_repeat': 256,
              'test_jets': 2000, 'balanced_classes': True, 'sampling_seeds': SEEDS, 'test_sampling_seed': 73,
              'quantum': {'qubits': 4, 'map': 'zz_feature_map', 'reps': 2, 'entanglement': 'linear',
                          'backend': 'Qiskit Statevector on CPU', 'shots': None, 'noise_model': None},
              'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'qiskit']},
              'python': platform.python_version(), 'platform': platform.platform(),
              'sources': manifest['sources'], 'prepared_hashes': hashes, 'code_sha256': sha256(__file__),
              'total_seconds': time.perf_counter()-started, 'selections': provenance,
              'per_repeat': metrics, 'summary': summary,
              'limitations': ['Small balanced sample and four engineered features',
                  'Overlapping fitting samples; repeats are not independent full studies',
                  'Classical exact simulation with classical SVM optimization',
                  'No hardware, noise, quantum advantage, or comparison to full-input MATLAB models']}
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    with (destination/'test_scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0])); writer.writeheader(); writer.writerows(score_rows)
    with (destination/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    plot(summary, destination)
    print(json.dumps(verify(destination)))
    print(json.dumps(summary, indent=2))


def plot(summary, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.bar([r['model'] for r in summary], [r['test_auc_mean'] for r in summary],
           yerr=[r['test_auc_sd'] for r in summary], capsize=4, color=['#64748b', '#2878a0', '#9354a6'])
    ax.set(ylim=(0, 1), ylabel='Test AUC', title='Four-feature pilot: classical and simulated quantum kernels')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .02, '512 training jets · 2,000 shared test jets · mean ± SD across 3 sampling repeats', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(Path(directory)/'kernel_comparison.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only: print(json.dumps(verify(args.output)))
    elif args.data is None: parser.error('--data is required for an experiment')
    else: run(args.data, args.output)
