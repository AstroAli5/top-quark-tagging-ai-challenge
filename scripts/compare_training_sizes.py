"""Verify and compare the fixed 50k/100k studies on their three shared seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from summarize_experiment import auc_influences, seed_interval
from verify_seed_extension import (BASELINE, ORIGINAL_TRAINING, EVALUATION_PATHS,
                                   configuration, fingerprint, git, semantic_manifest)

SEEDS = [101, 202, 303]
MODELS = ['CNN', 'GraphSAGE']
TRAIN_PATHS = ['src/core', 'src/pipeline', 'projectConfig.m', 'setupProject.m',
               'scripts/convert_dataset.py', 'scripts/download_dataset.py']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(baseline_report, larger_report, baseline_runs, larger_runs, output):
    reports = [json.loads(Path(p).read_text()) for p in [baseline_report, larger_report]]
    for report, size in zip(reports, [50000, 100000]):
        protocol = report['protocol']
        if (protocol['train_jets'], protocol['validation_jets'], protocol['test_jets'],
                protocol['epochs'], protocol['training_seeds']) != (size, 10000, 404000, 12, SEEDS):
            raise ValueError('Expected the fixed three-seed 50k/100k studies')
    if reports[0]['protocol']['sources'] != reports[1]['protocol']['sources']:
        raise ValueError('Source datasets differ')

    expected_train = fingerprint(BASELINE, TRAIN_PATHS)
    expected_eval = fingerprint(BASELINE, EVALUATION_PATHS)
    controllers = {git('rev-parse', f'{c}:run_experiment.m').strip()
                   for c in [ORIGINAL_TRAINING, BASELINE]}
    common_labels = common_rows = common_cfg = common_manifest = common_matlab = None
    records, checked = [], []
    for root, report, size in zip([baseline_runs, larger_runs], reports, [50000, 100000]):
        paths = list(Path(root).rglob('metadata.json'))
        selected = [(p, json.loads(p.read_text())) for p in paths]
        selected = [(p, m) for p, m in selected if m['trainingSeed'] in SEEDS]
        if sorted(m['trainingSeed'] for _, m in selected) != SEEDS:
            raise ValueError('Missing or duplicate matched runs')
        for path, m in sorted(selected, key=lambda item: item[1]['trainingSeed']):
            seed = m['trainingSeed']
            cfg = configuration(m['configuration'])
            manifest = semantic_manifest(m['manifest'])
            if manifest.pop('train_count') != size or m['models'] != MODELS:
                raise ValueError('Run size or model list disagrees with the study')
            if m['configuration']['cnnSeed'] != seed or m['configuration']['graphSeed'] != seed:
                raise ValueError('Training seed configuration disagrees')
            if common_cfg is None:
                common_cfg, common_manifest, common_matlab = cfg, manifest, m['matlabVersion']
            if cfg != common_cfg or manifest != common_manifest or m['matlabVersion'] != common_matlab:
                raise ValueError('Settings, validation/test selections or MATLAB version differ')
            train_commit = m.get('trainingProvenance', {}).get('codeCommit', m['codeCommit'])
            train_hash = fingerprint(train_commit, TRAIN_PATHS)
            eval_hash = fingerprint(m['codeCommit'], EVALUATION_PATHS)
            controller = git('rev-parse', f'{train_commit}:run_experiment.m').strip()
            if train_hash != expected_train or eval_hash != expected_eval or controller not in controllers:
                raise ValueError('Model implementation changed between training sizes')
            pred_path = path.parent / 'clean_predictions.mat'
            pred = loadmat(pred_path)
            labels, rows = pred['labels'].ravel(), pred['rows'].ravel()
            if (pred['probabilities'].shape != (404000, 2) or len(labels) != 404000
                    or not np.array_equal(rows, np.arange(404000)) or int(pred['seed'].item()) != seed):
                raise ValueError('Prediction coverage or seed disagrees')
            if common_labels is None:
                common_labels, common_rows = labels, rows
            if not np.array_equal(labels, common_labels) or not np.array_equal(rows, common_rows):
                raise ValueError('The studies evaluated different test rows or labels')
            for j, model in enumerate(MODELS):
                scores = pred['probabilities'][:, j]
                accuracy = float(np.mean((scores >= .5) == labels))
                auc = auc_influences(scores, labels)[0]
                saved = [r for r in report['per_seed_clean'] if r['Seed'] == seed and r['Model'] == model]
                if len(saved) != 1 or not np.allclose([accuracy, auc],
                        [saved[0]['Accuracy'], saved[0]['AUC']], atol=1e-10, rtol=0):
                    raise ValueError('Report values disagree with saved predictions')
                records.append({'TrainJets': size, 'Seed': seed, 'Model': model,
                                'Accuracy': accuracy, 'AUC': auc})
            checked.append({'train_jets': size, 'seed': seed, 'test_rows_verified': len(rows),
                            'workflow_run': m['workflowRun'], 'training_commit': train_commit,
                            'evaluation_commit': m['codeCommit'],
                            'training_source_sha256': train_hash, 'evaluation_source_sha256': eval_hash,
                            'predictions_sha256': sha256(pred_path), 'metadata_sha256': sha256(path)})

    frame = pd.DataFrame(records)
    differences, paired = [], []
    for model in MODELS:
        subset = frame[frame.Model == model]
        small = subset[subset.TrainJets == 50000].set_index('Seed')
        large = subset[subset.TrainJets == 100000].set_index('Seed')
        delta = large[['Accuracy', 'AUC']] - small[['Accuracy', 'AUC']]
        for seed in SEEDS:
            paired.append({'Seed': seed, 'Model': model, 'Accuracy_Difference': delta.loc[seed, 'Accuracy'],
                           'AUC_Difference': delta.loc[seed, 'AUC']})
        for metric in ['Accuracy', 'AUC']:
            mean, sd, low, high = seed_interval(delta[metric])
            differences.append({'Model': model, 'Metric': metric, 'Mean_Difference': mean,
                                'SeedSD': sd, 'SeedCI_Low': low, 'SeedCI_High': high})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'analysis_script_sha256': sha256(__file__),
              'input_report_sha256': [sha256(p) for p in [baseline_report, larger_report]],
              'training_seeds': SEEDS, 'verification': checked,
              'matched_per_seed': records, 'paired_differences': paired, 'summary': differences,
              'note': '100k minus 50k; three matched seed labels, identical test rows and model settings. '
                      'Equal epochs increase optimizer updates. Seed intervals condition on this test set; '
                      'this is not an architecture ablation or an equal-compute comparison.'}
    (output / 'training_size_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.DataFrame(paired).to_csv(output / 'training_size_paired.csv', index=False)
    pd.DataFrame(differences).to_csv(output / 'training_size_summary.csv', index=False)
    plot(frame, output)
    print(json.dumps(differences, indent=2))


def plot(frame, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    for ax, metric, scale, label in zip(axes, ['Accuracy', 'AUC'], [100, 1], ['Accuracy (%)', 'AUC']):
        for model, color in zip(MODELS, ['#2878a0', '#bf6434']):
            stats = frame[frame.Model == model].groupby('TrainJets')[metric].agg(['mean', 'std'])
            ax.errorbar([50, 100], stats['mean'] * scale, yerr=stats['std'] * scale,
                        color=color, marker='o', capsize=4, label=model)
        ax.set(xlabel='Official training jets (thousands)', ylabel=label, xticks=[50, 100], xlim=(43, 107))
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('50k to 100k training: same three seeds, 404,000 test jets')
    fig.text(.5, .01, 'Mean ± training-seed SD · 12 epochs at both sizes means more updates at 100k', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(output / 'training_size_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--larger-report', type=Path, required=True)
    parser.add_argument('--baseline-runs', type=Path, required=True)
    parser.add_argument('--larger-runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline_report, args.larger_report, args.baseline_runs, args.larger_runs, args.output)
