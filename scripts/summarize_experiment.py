"""Summarize fixed-model test uncertainty separately from training-seed variation."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import t

MODELS=['CNN','GraphSAGE','ResNeXt-SE reference']


def auc_influences(scores, labels):
    """Tie-aware AUC and per-positive/per-negative placement values."""
    scores=np.asarray(scores,float).reshape(-1)
    labels=np.asarray(labels).reshape(-1)
    if len(scores)!=len(labels) or not np.isfinite(scores).all() or not np.isin(labels,[0,1]).all():
        raise ValueError('Invalid scores or labels')
    positive=scores[labels==1]; negative=scores[labels==0]
    if min(len(positive),len(negative))<2:
        raise ValueError('At least two jets per class are required')
    a=np.sort(positive); b=np.sort(negative)
    vp=(np.searchsorted(b,positive,'left')+np.searchsorted(b,positive,'right'))/(2*len(b))
    vn=1-(np.searchsorted(a,negative,'left')+np.searchsorted(a,negative,'right'))/(2*len(a))
    return float(vp.mean()),vp,vn


def seed_interval(values):
    values=np.asarray(values,float)
    mean=float(values.mean()); sd=float(values.std(ddof=1))
    half=float(t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)))
    return mean,sd,mean-half,mean+half


def verify_saved_run(clean, noisy, metadata, predictions, noise, models=MODELS):
    """Check coverage and paired noise, allowing only saved-score rounding error."""
    seed = int(metadata['trainingSeed'])
    size = metadata['manifest']['test_count']
    labels = predictions['labels'].ravel()
    scores = predictions['probabilities']
    if (len(labels) != size or scores.shape != (size, len(models))
            or not np.array_equal(predictions['rows'].ravel(), np.arange(size))
            or size != metadata['manifest']['sources']['test']['available_rows']):
        raise ValueError('Saved predictions do not cover every official test row')
    if (int(predictions['seed'].item()) != seed or set(clean.Seed) != {seed}
            or set(noisy.Seed) != {seed} or len(clean) != len(models)
            or set(clean.Model) != set(models) or not (clean.TestJets == size).all()):
        raise ValueError('Prediction seed or clean metric rows disagree')
    cfg = metadata['configuration']
    levels = np.asarray(cfg['noiseLevels']); seeds = np.asarray(cfg['noiseSeeds'])
    count = min(cfg['noiseTestJets'], size)
    saved = noise['noisePredictions']
    if (saved.shape != (count, len(models), len(levels), len(seeds))
            or not np.array_equal(noise['noiseLabels'].ravel(), labels[:count])
            or not np.array_equal(noise['noiseLevels'].ravel(), levels)
            or not np.array_equal(noise['noiseSeeds'].ravel(), seeds)
            or int(noise['seed'].item()) != seed):
        raise ValueError('Noise predictions do not match their declared sample')
    if (not np.isfinite(scores).all() or not np.isfinite(saved).all()
            or (scores < 0).any() or (scores > 1).any()
            or (saved < 0).any() or (saved > 1).any()):
        raise ValueError('Invalid saved probabilities')
    expected = {(float(level), int(draw), model) for level in levels for draw in seeds for model in models}
    actual = set(zip(noisy.Sigma, noisy.NoiseSeed, noisy.Model))
    if len(noisy) != len(expected) or actual != expected or not (noisy.TestJets == count).all():
        raise ValueError('Noise metric rows are missing or duplicated')
    y = labels[:count].astype(bool)
    largest_difference = 0.
    for l, level in enumerate(levels):
        for r, draw in enumerate(seeds):
            if level == 0 and not np.array_equal(saved[:, :, l, r], scores[:count].astype(saved.dtype)):
                raise ValueError('Zero-noise predictions differ from the same clean jets')
            for j, model in enumerate(models):
                p = saved[:, j, l, r]
                row = noisy[(noisy.Sigma == level) & (noisy.NoiseSeed == draw) & (noisy.Model == model)].iloc[0]
                # MATLAB saves noise scores as float32. Bracket the original
                # score by adjacent representable values instead of imposing
                # an arbitrary AUC tolerance near ties or the 0.5 threshold.
                low = np.nextafter(p, -np.inf).astype(float)
                high = np.nextafter(p, np.inf).astype(float)
                accuracy_low = np.mean(np.where(y, low >= .5, high < .5))
                accuracy_high = np.mean(np.where(y, high >= .5, low < .5))
                def pair_auc(positive, negative):
                    negative = np.sort(negative)
                    return float(np.mean((np.searchsorted(negative, positive, 'left')
                        + np.searchsorted(negative, positive, 'right')) / (2 * len(negative))))
                auc_low = pair_auc(low[y], high[~y]); auc_high = pair_auc(high[y], low[~y])
                if (not accuracy_low - 1e-10 <= row.Accuracy <= accuracy_high + 1e-10
                        or not auc_low - 1e-10 <= row.AUC <= auc_high + 1e-10):
                    raise ValueError('Noise metrics disagree with saved predictions beyond rounding')
                largest_difference = max(largest_difference, abs(auc_influences(p, y)[0] - row.AUC))
    return {'training_seed': seed, 'test_rows_verified': size, 'noise_rows_verified': count,
            'noise_metric_rows_verified': len(noisy), 'maximum_noise_auc_rounding_difference': largest_difference}


def summarize(source, destination):
    source=Path(source); destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    paths=sorted(source.rglob('clean_metrics.csv'))
    if len(paths)!=3:
        raise ValueError(f'Expected exactly three complete training runs; found {len(paths)}')
    clean=[]; noisy=[]; metadata=[]; influences=[]; verification=[]; common_labels=None; common_rows=None; models=None
    for path in paths:
        folder=path.parent
        c=pd.read_csv(path); n=pd.read_csv(folder/'noise_metrics.csv')
        m=json.loads((folder/'metadata.json').read_text())
        current_models=m.get('models',MODELS)
        if models is None: models=current_models
        if current_models!=models: raise ValueError('Runs evaluated different models')
        predictions=loadmat(folder/'clean_predictions.mat')
        verification.append(verify_saved_run(c,n,m,predictions,loadmat(folder/'noise_predictions.mat'),models))
        labels=predictions['labels'].ravel(); rows=predictions['rows'].ravel()
        if common_labels is None:
            common_labels=labels; common_rows=rows
        elif not np.array_equal(common_labels,labels) or not np.array_equal(common_rows,rows):
            raise ValueError('Training runs did not evaluate identical official test rows')
        values=[]
        for j,model in enumerate(models):
            auc,vp,vn=auc_influences(predictions['probabilities'][:,j],labels)
            reported=c.loc[c.Model==model,'AUC'].item()
            accuracy=float(np.mean((predictions['probabilities'][:,j]>=.5)==labels))
            if not np.isclose(auc,reported,atol=1e-10,rtol=0) or not np.isclose(accuracy,c.loc[c.Model==model,'Accuracy'].item(),atol=1e-10,rtol=0):
                raise ValueError('Saved predictions disagree with reported metrics')
            variance=vp.var(ddof=1)/len(vp)+vn.var(ddof=1)/len(vn)
            c.loc[c.Model==model,'AUC_TestCI_Low']=max(0,auc-1.96*np.sqrt(variance))
            c.loc[c.Model==model,'AUC_TestCI_High']=min(1,auc+1.96*np.sqrt(variance))
            # Wilson interval for accuracy on this fixed test set/model.
            size=len(labels); center=(accuracy+1.96**2/(2*size))/(1+1.96**2/size)
            half=1.96*np.sqrt(accuracy*(1-accuracy)/size+1.96**2/(4*size**2))/(1+1.96**2/size)
            c.loc[c.Model==model,'Accuracy_TestCI_Low']=center-half
            c.loc[c.Model==model,'Accuracy_TestCI_High']=center+half
            values.append((vp,vn))
        clean.append(c); noisy.append(n); metadata.append(m); influences.append(values)
    clean=pd.concat(clean,ignore_index=True); noisy=pd.concat(noisy,ignore_index=True)
    if set(clean.Seed)!={101,202,303}:
        raise ValueError('The frozen training seeds must be 101, 202, 303')
    first=metadata[0]
    signature=first['manifest']['fitting_sha256']
    for m in metadata:
        if m['manifest']!=first['manifest'] or m['codeCommit']!=first['codeCommit']:
            raise ValueError('The runs used different data or code versions')
        if m.get('evaluationProvenance')!=first.get('evaluationProvenance'):
            raise ValueError('Training seeds used different family evaluation versions')
        if not m['manifest']['full_official_test']:
            raise ValueError('The full official test partition was not evaluated')
        ignored={'dataDir','modelsDir','resultsDir','cnnSeed','graphSeed','winnerSeed'}
        settings=lambda config: {k:v for k,v in config.items() if k not in ignored}
        if settings(m['configuration']) != settings(first['configuration']):
            raise ValueError('Training runs used different experiment settings')
    means=[]
    for model in models:
        row={'Model':model}
        for metric in ['Accuracy','AUC']:
            mean,sd,low,high=seed_interval(clean.loc[clean.Model==model,metric])
            row.update({f'{metric}_Mean':mean,f'{metric}_SeedSD':sd,
                        f'{metric}_SeedCI_Low':low,f'{metric}_SeedCI_High':high})
        row['TrainingSeconds_Mean']=float(clean.loc[clean.Model==model,'TrainingSeconds'].mean())
        means.append(row)
    clean_summary=pd.DataFrame(means)
    # Repeated noise draws share one fitted model. Average within each training
    # seed before computing between-training-seed SD; do not treat all nine as independent.
    by_seed=noisy.groupby(['Seed','Sigma','Model'],as_index=False)[['Accuracy','AUC']].mean()
    noise_summary=by_seed.groupby(['Sigma','Model']).agg(
        Accuracy_Mean=('Accuracy','mean'),Accuracy_SeedSD=('Accuracy','std'),
        AUC_Mean=('AUC','mean'),AUC_SeedSD=('AUC','std')).reset_index()
    paired=[]
    for a,b in itertools.combinations(range(len(models)),2):
        left=clean[clean.Model==models[a]].set_index('Seed').AUC
        right=clean[clean.Model==models[b]].set_index('Seed').AUC
        mean,sd,low,high=seed_interval(right-left)
        vp=np.mean([run[b][0]-run[a][0] for run in influences],axis=0)
        vn=np.mean([run[b][1]-run[a][1] for run in influences],axis=0)
        se=np.sqrt(vp.var(ddof=1)/len(vp)+vn.var(ddof=1)/len(vn))
        paired.append({'Difference':f'{models[b]} minus {models[a]}','MeanAUC_Difference':mean,
            'TrainingSeedSD':sd,'TrainingSeedCI_Low':low,'TrainingSeedCI_High':high,
            'ConditionalTestCI_Low':mean-1.96*se,'ConditionalTestCI_High':mean+1.96*se})
    report={'code_commit':first['codeCommit'],'workflow_run':first['workflowRun'],
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fitting_sha256':signature,'verification':verification,'models':models,
        'training_provenance':[m.get('trainingProvenance',{'codeCommit':m['codeCommit']}) for m in metadata],
        'evaluation_provenance':first.get('evaluationProvenance',{'codeCommit':first['codeCommit'],'workflowRun':first['workflowRun']}),
        'protocol':{'train_jets':first['manifest']['train_count'],'validation_jets':first['manifest']['val_count'],
            'test_jets':len(common_labels),'noise_test_jets':int(noisy.TestJets.iloc[0]),
            'training_seeds':[101,202,303],'noise_seeds':first['configuration']['noiseSeeds'],
            'epochs':first['configuration']['cnnEpochs'],'sources':first['manifest']['sources']},
        'clean_summary':means,'per_seed_clean':clean.to_dict(orient='records'),
        'noise_summary':noise_summary.to_dict(orient='records'),
        'per_seed_noise':noisy.to_dict(orient='records'),'paired_auc':paired,
        'uncertainty_note':'Seed intervals describe training variability on a fixed test set. Conditional test intervals hold fitted models fixed. Neither covers detector-model mismatch.'}
    clean.to_csv(destination/'per_seed_clean.csv',index=False)
    noisy.to_csv(destination/'per_seed_noise.csv',index=False)
    clean_summary.to_csv(destination/'clean_summary.csv',index=False)
    noise_summary.to_csv(destination/'noise_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(destination/'paired_auc.csv',index=False)
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    make_plots(report,destination)
    print('REPORT_JSON_START')
    print(json.dumps(report))
    print('REPORT_JSON_END')


def make_plots(report,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    clean=pd.DataFrame(report['clean_summary']); noise=pd.DataFrame(report['noise_summary'])
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    palette=dict(zip(MODELS,['#2878a0','#bf6434','#637b4b']))
    models=clean.Model.tolist(); colors=[palette[model] for model in models]
    for color,model in zip(colors,models):
        d=noise[noise.Model==model].sort_values('Sigma')
        ax.errorbar(d.Sigma,d.AUC_Mean,yerr=d.AUC_SeedSD,label=model,color=color,marker='o',capsize=3)
    ax.set(xlabel='Synthetic component smearing (fraction)',ylabel='AUC',title='Paired noise study: mean and SD across 3 training seeds')
    ax.legend(frameon=False); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(Path(destination)/'noise_auc.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    ax.bar(clean.Model,clean.Accuracy_Mean*100,yerr=clean.Accuracy_SeedSD*100,color=colors,capsize=4)
    ax.set(ylabel='Accuracy (%)',ylim=(0,100),title='Full official test set: mean and SD across 3 training seeds')
    fig.tight_layout(); fig.savefig(Path(destination)/'clean_accuracy.png',dpi=180); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); summarize(args.input,args.output)
