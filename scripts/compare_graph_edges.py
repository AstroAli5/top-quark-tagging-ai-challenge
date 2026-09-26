"""Verify the three-seed 100k graph-neighbor ablation against its saved baseline."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from summarize_experiment import auc_influences, seed_interval, verify_saved_run
from verify_seed_extension import fingerprint, semantic_manifest, configuration

BASELINE = '0dec3b285672c5e6f8eb3bc89c1d646505387642'
SOURCE_PATHS = ['src/core', 'src/pipeline', 'src/experiment/evaluateOfficialTest.m',
                'src/experiment/predictThreeModels.m', 'src/experiment/readJetChunk.m',
                'projectConfig.m', 'setupProject.m', 'run_experiment.m',
                'scripts/prepare_official.py', 'scripts/convert_dataset.py', 'scripts/download_dataset.py']
SEEDS = [101, 202, 303]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(baseline, variant, output):
    source_hash = fingerprint(BASELINE, SOURCE_PATHS)
    common_cfg = common_manifest = common_labels = common_rows = common_matlab = None
    clean_rows, noise_rows, evidence = [], [], []
    for directory, name, neighbors, models in [(baseline, 'GraphSAGE (k=6)', 6, ['CNN','GraphSAGE']),
                                               (variant, 'No neighbors (k=0)', 0, ['GraphSAGE'])]:
        inputs = [(p, json.loads(p.read_text())) for p in Path(directory).rglob('metadata.json')]
        if sorted(m['trainingSeed'] for _,m in inputs) != SEEDS:
            raise ValueError('Expected exactly three runs with seeds 101/202/303')
        for path, meta in sorted(inputs, key=lambda item:item[1]['trainingSeed']):
            folder = path.parent
            seed = meta['trainingSeed']
            cfg = configuration(meta['configuration'])
            if cfg.pop('kNeighbors') != neighbors or cfg.pop('experimentModels') != models or meta['models'] != models:
                raise ValueError('Wrong graph control or selected model')
            if meta['configuration']['graphSeed'] != seed:
                raise ValueError('Configured training seed disagrees')
            manifest = semantic_manifest(meta['manifest'])
            if (manifest['train_count'],manifest['val_count'],manifest['test_count']) != (100000,10000,404000):
                raise ValueError('Wrong fitting or evaluation size')
            if cfg['graphEpochs'] != 12 or not manifest['full_official_test']:
                raise ValueError('Wrong epoch budget or test coverage')
            if common_cfg is None:
                common_cfg, common_manifest, common_matlab = cfg, manifest, meta['matlabVersion']
            if cfg != common_cfg or manifest != common_manifest or meta['matlabVersion'] != common_matlab:
                raise ValueError('Configuration, source selections or MATLAB version changed beyond the control')
            if fingerprint(meta['codeCommit'],SOURCE_PATHS) != source_hash:
                raise ValueError('Trainer, preparer or evaluator differs from the specified baseline')
            clean = pd.read_csv(folder/'clean_metrics.csv')
            noisy = pd.read_csv(folder/'noise_metrics.csv')
            predictions = loadmat(folder/'clean_predictions.mat')
            noise = loadmat(folder/'noise_predictions.mat')
            verified = verify_saved_run(clean,noisy,meta,predictions,noise,models)
            labels, rows = predictions['labels'].ravel(), predictions['rows'].ravel()
            if common_labels is None:
                common_labels, common_rows = labels, rows
            if not np.array_equal(labels,common_labels) or not np.array_equal(rows,common_rows):
                raise ValueError('The controls evaluated different test rows or labels')
            for j, model in enumerate(models):
                score = predictions['probabilities'][:,j]
                accuracy = float(np.mean((score>=.5)==labels))
                auc = auc_influences(score,labels)[0]
                record = clean[clean.Model==model].iloc[0]
                if not np.allclose([accuracy,auc],[record.Accuracy,record.AUC],rtol=0,atol=1e-10):
                    raise ValueError('Clean metrics disagree with saved predictions')
                if model == 'GraphSAGE':
                    clean_rows.append({'Variant':name,'Seed':seed,'Accuracy':accuracy,'AUC':auc,
                                       'TrainingSeconds':float(record.TrainingSeconds)})
            graph_noise = noisy[noisy.Model=='GraphSAGE'].copy()
            graph_noise['Variant'] = name
            noise_rows.extend(graph_noise.drop(columns=['Model']).to_dict(orient='records'))
            checkpoint = folder.parent/'models/graphsage_model.mat'
            checkpoint_cfg = loadmat(checkpoint,variable_names=['cfg'],simplify_cells=True)['cfg']
            if checkpoint_cfg['kNeighbors'] != neighbors or checkpoint_cfg['graphSeed'] != seed:
                raise ValueError('Checkpoint configuration disagrees with the ablation label')
            evidence.append({'variant':name,'seed':seed,'workflow_run':meta['workflowRun'],
                             'code_commit':meta['codeCommit'],'source_sha256':source_hash,
                             'checkpoint_sha256':sha256(checkpoint),'metadata_sha256':sha256(path),
                             'clean_predictions_sha256':sha256(folder/'clean_predictions.mat'),
                             'noise_predictions_sha256':sha256(folder/'noise_predictions.mat'),
                             'verification':verified})
    clean = pd.DataFrame(clean_rows)
    noisy = pd.DataFrame(noise_rows)
    summary = []
    for variant_name, frame in clean.groupby('Variant',sort=False):
        record = {'Variant':variant_name}
        for metric in ['Accuracy','AUC']:
            mean,sd,low,high = seed_interval(frame[metric])
            record.update({metric+'_Mean':mean,metric+'_SeedSD':sd,
                           metric+'_SeedCI_Low':low,metric+'_SeedCI_High':high})
        summary.append(record)
    baseline_clean = clean[clean.Variant=='GraphSAGE (k=6)'].set_index('Seed')
    variant_clean = clean[clean.Variant=='No neighbors (k=0)'].set_index('Seed')
    paired = (variant_clean[['Accuracy','AUC']]-baseline_clean[['Accuracy','AUC']]).reset_index()
    differences = []
    for metric in ['Accuracy','AUC']:
        mean,sd,low,high = seed_interval(paired[metric])
        differences.append({'Metric':metric,'Mean_Difference':mean,'SeedSD':sd,
                            'SeedCI_Low':low,'SeedCI_High':high})
    noise_by_seed = noisy.groupby(['Variant','Seed','Sigma'],as_index=False)[['Accuracy','AUC']].mean()
    noise_summary = noise_by_seed.groupby(['Variant','Sigma'],as_index=False).agg(
        AUC_Mean=('AUC','mean'),AUC_SeedSD=('AUC','std'),Accuracy_Mean=('Accuracy','mean'))
    noise_differences = []
    for sigma in sorted(noise_by_seed.Sigma.unique()):
        pair = noise_by_seed[noise_by_seed.Sigma==sigma].pivot(index='Seed',columns='Variant',values='AUC')
        mean,sd,low,high = seed_interval(pair['No neighbors (k=0)']-pair['GraphSAGE (k=6)'])
        noise_differences.append({'Sigma':float(sigma),'Mean_AUC_Difference':mean,'SeedSD':sd,
                                  'SeedCI_Low':low,'SeedCI_High':high})
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    report = {'analysis_script_sha256':sha256(__file__),'verified_source_commit':BASELINE,
              'protocol':{'train_jets':100000,'validation_jets':10000,'test_jets':404000,
                          'training_seeds':SEEDS,'epochs':12,'noise_test_jets':10000},
              'verification':evidence,'clean_summary':summary,'clean_paired_differences':differences,
              'noise_paired_differences':noise_differences,
              'difference_direction':'No neighbors (k=0) minus GraphSAGE (k=6)',
              'interpretation':'Only access to neighbor messages changes. Effective capacity differs. '
                 'Three paired seed labels and a fixed test set; no CNN-versus-graph architecture claim.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for filename,frame in [('per_seed_clean',clean),('per_seed_noise',noisy),('clean_summary',pd.DataFrame(summary)),
                          ('clean_paired',paired),('noise_summary',noise_summary),
                          ('noise_paired',pd.DataFrame(noise_differences))]:
        frame.to_csv(output/(filename+'.csv'),index=False)
    plot(pd.DataFrame(summary),noise_summary,output)
    print(json.dumps(report,indent=2))


def plot(clean,noise,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(10,4.6))
    colors = ['#bf6434','#637b4b']
    axes[0].bar(['Neighbors (k=6)','No neighbors (k=0)'],clean.AUC_Mean,
                yerr=clean.AUC_SeedSD,color=colors,capsize=4)
    axes[0].set(ylabel='Full official test AUC',ylim=(0,1))
    for name,color in zip(clean.Variant,colors):
        frame = noise[noise.Variant==name].sort_values('Sigma')
        axes[1].errorbar(frame.Sigma,frame.AUC_Mean,yerr=frame.AUC_SeedSD,
                         label=name,color=color,marker='o',capsize=3)
    axes[1].set(xlabel='Synthetic component smearing',ylabel='AUC on 10,000 test jets')
    axes[1].legend(frameon=False,fontsize=9); axes[1].grid(alpha=.2)
    fig.suptitle('Graph-neighbor control: 100k training jets, three seeds')
    fig.text(.5,.01,'Mean ± training-seed SD · identical node features and training settings',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.95)); fig.savefig(output/'graph_edge_comparison.png',dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--variant',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); compare(args.baseline,args.variant,args.output)
