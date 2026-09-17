"""Combine separately evaluated model families only after checking identical jets."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from summarize_experiment import MODELS, verify_saved_run


def combine(source, destination):
    source=Path(source); destination=Path(destination)
    for seed in [101,202,303]:
        folders=[source/f'research-{family}-seed-{seed}'/'results' for family in ['core','reference']]
        meta=[json.loads((p/'metadata.json').read_text()) for p in folders]
        clean=[pd.read_csv(p/'clean_metrics.csv') for p in folders]
        noisy=[pd.read_csv(p/'noise_metrics.csv') for p in folders]
        predictions=[loadmat(p/'clean_predictions.mat') for p in folders]
        noise=[loadmat(p/'noise_predictions.mat') for p in folders]
        assert meta[0]['models']==MODELS[:2] and meta[1]['models']==MODELS[2:], 'Unexpected model families'
        for i in range(2):
            verify_saved_run(clean[i],noisy[i],meta[i],predictions[i],noise[i],meta[i]['models'])
            assert meta[i]['trainingSeed']==seed, 'Wrong training seed'
        assert meta[0]['manifest']==meta[1]['manifest'], 'Different prepared data'
        if meta[0]['codeCommit'] != meta[1]['codeCommit']:
            revisions=[item['codeCommit'] for item in meta]
            if not all(re.fullmatch('[0-9a-f]{40}',revision) for revision in revisions):
                raise ValueError('Evaluation revisions must be full Git commit hashes')
            # Separate recovery commits may add orchestration or documentation;
            # the model, representation, and evaluation code must be unchanged.
            shared=['src/core','src/reference','src/experiment/evaluateOfficialTest.m',
                    'src/experiment/predictThreeModels.m','src/experiment/readJetChunk.m',
                    'run_experiment.m','projectConfig.m']
            subprocess.run(['git','diff','--exit-code',*revisions,'--',*shared],
                cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True)
        for key in ['labels','rows']:
            np.testing.assert_array_equal(predictions[0][key],predictions[1][key])
        for key in ['noiseLabels','noiseLevels','noiseSeeds']:
            np.testing.assert_array_equal(noise[0][key],noise[1][key])
        ignored={'dataDir','modelsDir','resultsDir','inputFile','experimentModels',
                 'recoveredFromWorkflow','trainingCommit','trainingTimeSource'}
        settings=lambda item: {k:v for k,v in item['configuration'].items() if k not in ignored}
        assert settings(meta[0])==settings(meta[1]), 'Different experiment settings'
        out=destination/f'seed_{seed}'/'results'; out.mkdir(parents=True,exist_ok=False)
        pd.concat(clean,ignore_index=True).to_csv(out/'clean_metrics.csv',index=False)
        pd.concat(noisy,ignore_index=True).to_csv(out/'noise_metrics.csv',index=False)
        p={key:predictions[0][key] for key in ['labels','rows','seed']}
        p['probabilities']=np.concatenate([v['probabilities'] for v in predictions],axis=1)
        savemat(out/'clean_predictions.mat',p,do_compression=True)
        n={key:noise[0][key] for key in ['noiseLabels','noiseLevels','noiseSeeds','seed']}
        n['noisePredictions']=np.concatenate([v['noisePredictions'] for v in noise],axis=1)
        savemat(out/'noise_predictions.mat',n,do_compression=True)
        m=meta[0].copy(); m['models']=MODELS
        m['configuration']=m['configuration'].copy(); m['configuration']['experimentModels']=MODELS
        m['normalization']=meta[1]['normalization']
        m['trainingProvenance']={family:item.get('trainingProvenance',{'codeCommit':item['codeCommit'],
            'workflowRun':item['workflowRun']}) for family,item in zip(['core','reference'],meta)}
        m['familyDatasetIds']={family:item['datasetId'] for family,item in zip(['core','reference'],meta)}
        m['evaluationProvenance']={family:{'codeCommit':item['codeCommit'],'workflowRun':item['workflowRun']}
            for family,item in zip(['core','reference'],meta)}
        (out/'metadata.json').write_text(json.dumps(m,indent=2)+'\n')
    print('Combined all three seeds after verifying source hashes, settings, rows, and predictions.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); combine(args.input,args.output)
