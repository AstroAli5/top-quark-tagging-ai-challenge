"""Summarize fixed-model test uncertainty separately from training-seed variation."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm, t

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


def summarize(source, destination):
    source=Path(source); destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    paths=sorted(source.rglob('clean_metrics.csv'))
    if len(paths)!=3:
        raise ValueError(f'Expected exactly three complete training runs; found {len(paths)}')
    clean=[]; noisy=[]; metadata=[]; influences=[]; common_labels=None; common_rows=None
    for path in paths:
        folder=path.parent
        c=pd.read_csv(path); n=pd.read_csv(folder/'noise_metrics.csv')
        m=json.loads((folder/'metadata.json').read_text())
        predictions=loadmat(folder/'clean_predictions.mat')
        labels=predictions['labels'].ravel(); rows=predictions['rows'].ravel()
        if common_labels is None:
            common_labels=labels; common_rows=rows
        elif not np.array_equal(common_labels,labels) or not np.array_equal(common_rows,rows):
            raise ValueError('Training runs did not evaluate identical official test rows')
        values=[]
        for j,model in enumerate(MODELS):
            auc,vp,vn=auc_influences(predictions['probabilities'][:,j],labels)
            reported=c.loc[c.Model==model,'AUC'].item()
            accuracy=float(np.mean((predictions['probabilities'][:,j]>=.5)==labels))
            if not np.isclose(auc,reported,atol=1e-10) or not np.isclose(accuracy,c.loc[c.Model==model,'Accuracy'].item(),atol=1e-10):
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
        if m['manifest']['fitting_sha256']!=signature or m['codeCommit']!=first['codeCommit']:
            raise ValueError('The runs used different data or code versions')
        if not m['manifest']['full_official_test']:
            raise ValueError('The full official test partition was not evaluated')
    means=[]
    for model in MODELS:
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
    for a,b in itertools.combinations(range(3),2):
        left=clean[clean.Model==MODELS[a]].set_index('Seed').AUC
        right=clean[clean.Model==MODELS[b]].set_index('Seed').AUC
        mean,sd,low,high=seed_interval(right-left)
        vp=np.mean([run[b][0]-run[a][0] for run in influences],axis=0)
        vn=np.mean([run[b][1]-run[a][1] for run in influences],axis=0)
        se=np.sqrt(vp.var(ddof=1)/len(vp)+vn.var(ddof=1)/len(vn))
        paired.append({'Difference':f'{MODELS[b]} minus {MODELS[a]}','MeanAUC_Difference':mean,
            'TrainingSeedSD':sd,'TrainingSeedCI_Low':low,'TrainingSeedCI_High':high,
            'ConditionalTestCI_Low':mean-1.96*se,'ConditionalTestCI_High':mean+1.96*se})
    report={'code_commit':first['codeCommit'],'workflow_run':first['workflowRun'],
        'protocol':{'train_jets':first['manifest']['train_count'],'validation_jets':first['manifest']['val_count'],
            'test_jets':len(common_labels),'noise_test_jets':int(noisy.TestJets.iloc[0]),
            'training_seeds':[101,202,303],'noise_seeds':first['configuration']['noiseSeeds'],
            'epochs':first['configuration']['cnnEpochs'],'sources':first['manifest']['sources']},
        'clean_summary':means,'per_seed_clean':clean.to_dict(orient='records'),
        'noise_summary':noise_summary.to_dict(orient='records'),'paired_auc':paired,
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
    colors=['#2878a0','#bf6434','#637b4b']
    for color,model in zip(colors,MODELS):
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
