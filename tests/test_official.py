import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_official import prepare, read_rows
from convert_dataset import PARTICLE_COLUMNS
from summarize_experiment import MODELS, auc_influences, verify_saved_run
from summarize_experiment import summarize
from combine_study_runs import combine


class OfficialPartitions(unittest.TestCase):
    def test_separate_model_families_combine_and_summarize_consistently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); labels=np.array([0,1]*5); rows=np.arange(10)
            levels=np.array([0.,.1]); draws=np.array([7,17,27])
            for seed in [101,202,303]:
                for family,models in [('core',MODELS[:2]),('reference',MODELS[2:])]:
                    folder=root/'input'/f'research-{family}-seed-{seed}'/'results'
                    folder.mkdir(parents=True)
                    scores=np.random.default_rng(seed).uniform(.05,.95,(10,len(models))).astype('float32')
                    cfg=dict(cnnEpochs=12,graphEpochs=12,winnerEpochs=12,cnnSeed=seed,
                        graphSeed=seed,winnerSeed=seed,noiseLevels=levels.tolist(),noiseSeeds=draws.tolist(),
                        noiseTestJets=10,winnerWidths=[32,64,128],winnerGroups=4,winnerLearnRate=.005,
                        winnerImageSize=37,winnerMaxParticles=35,winnerBatchSize=64,experimentModels=models)
                    meta=dict(trainingSeed=seed,models=models,configuration=cfg,codeCommit='test-fixture',
                        workflowRun='synthetic',datasetId=f'{family}-{seed}',normalization={},
                        manifest=dict(test_count=10,train_count=20,val_count=10,full_official_test=True,
                            fitting_sha256='fixture',sources={'test':{'available_rows':10}}))
                    (folder/'metadata.json').write_text(json.dumps(meta))
                    clean=[]; noisy=[]
                    for j,model in enumerate(models):
                        accuracy=np.mean((scores[:,j]>=.5)==labels); auc=auc_influences(scores[:,j],labels)[0]
                        clean.append(dict(Seed=seed,Model=model,TestJets=10,Accuracy=accuracy,AUC=auc,TrainingSeconds=1.))
                        noisy.extend(dict(Seed=seed,Model=model,TestJets=10,Accuracy=accuracy,AUC=auc,
                            Sigma=float(level),NoiseSeed=int(draw)) for level in levels for draw in draws)
                    pd.DataFrame(clean).to_csv(folder/'clean_metrics.csv',index=False)
                    pd.DataFrame(noisy).to_csv(folder/'noise_metrics.csv',index=False)
                    savemat(folder/'clean_predictions.mat',dict(labels=labels,rows=rows,probabilities=scores,seed=seed))
                    savemat(folder/'noise_predictions.mat',dict(noiseLabels=labels,noiseLevels=levels,noiseSeeds=draws,
                        noisePredictions=np.tile(scores[:,:,None,None],(1,1,2,3)),seed=seed))
            with patch('builtins.print'), patch('summarize_experiment.make_plots'):
                combine(root/'input',root/'combined')
                summarize(root/'combined',root/'summary')
            report=json.loads((root/'summary/report.json').read_text())
            self.assertEqual(report['models'],MODELS)
            self.assertEqual(len(report['per_seed_clean']),9)
            self.assertEqual(len(report['per_seed_noise']),54)
            self.assertEqual(len(report['verification']),3)
            original=loadmat(root/'input/research-core-seed-101/results/clean_predictions.mat')
            combined=loadmat(root/'combined/seed_101/results/clean_predictions.mat')
            np.testing.assert_array_equal(combined['probabilities'][:,:2],original['probabilities'])
            bad=root/'input/research-reference-seed-101/results/metadata.json'
            changed=json.loads(bad.read_text()); changed['manifest']['fitting_sha256']='different'
            bad.write_text(json.dumps(changed))
            with self.assertRaisesRegex(AssertionError,'Different prepared data'):
                combine(root/'input',root/'invalid')

    def test_saved_noise_rounding_is_allowed_but_corrupt_metrics_are_rejected(self):
        labels = np.array([0,1,0,1,0,1])
        p = np.array([.1,.9,.7,.3,.500000001,.499999999])
        scores = np.tile(p[:,None],(1,3))
        levels = np.array([0.,.1]); draws = np.array([7,17,27])
        cfg = {'noiseLevels':levels.tolist(),'noiseSeeds':draws.tolist(),'noiseTestJets':6}
        meta = {'trainingSeed':101,'configuration':cfg,
                'manifest':{'test_count':6,'sources':{'test':{'available_rows':6}}}}
        predictions = {'labels':labels,'rows':np.arange(6),'probabilities':scores,'seed':np.array(101)}
        saved = np.tile(scores.astype('float32')[:,:,None,None],(1,1,2,3))
        noise = {'noisePredictions':saved,'noiseLabels':labels,'noiseLevels':levels,
                 'noiseSeeds':draws,'seed':np.array(101)}
        clean = pd.DataFrame({'Seed':[101]*3,'Model':MODELS,'TestJets':[6]*3})
        noisy = pd.DataFrame([{'Seed':101,'NoiseSeed':int(draw),'Sigma':level,'Model':model,
            'TestJets':6,'Accuracy':np.mean((p>=.5)==labels),'AUC':auc_influences(p,labels)[0]}
            for level in levels for draw in draws for model in MODELS])
        self.assertEqual(verify_saved_run(clean,noisy,meta,predictions,noise)['noise_metric_rows_verified'],18)
        changed = noisy.copy(); changed.loc[0,'AUC'] = 0.
        with self.assertRaisesRegex(ValueError,'beyond rounding'):
            verify_saved_run(clean,changed,meta,predictions,noise)
        with self.assertRaisesRegex(ValueError,'missing or duplicated'):
            verify_saved_run(clean,noisy.iloc[:-1],meta,predictions,noise)
        noise['noisePredictions'] = saved.copy(); noise['noisePredictions'][0,0,0,1] = .8
        with self.assertRaisesRegex(ValueError,'Zero-noise'):
            verify_saved_run(clean,noisy,meta,predictions,noise)

    def test_root_commands_cannot_shadow_implementation(self):
        root = Path(__file__).resolve().parents[1]
        duplicates = [p.name for p in (root/'src').rglob('*.m') if (root/p.name).exists()]
        self.assertEqual(duplicates, [], 'MATLAB searches the current folder before its path')

    def test_auc_uncertainty_uses_tie_aware_pairwise_placements(self):
        labels=np.array([0,1,0,1,1,0])
        scores=np.array([.1,.3,.3,.8,.8,.8])
        positive=scores[labels==1,None]; negative=scores[labels==0][None,:]
        expected=(positive>negative).astype(float)+.5*(positive==negative)
        auc,vp,vn=auc_influences(scores,labels)
        self.assertAlmostEqual(auc,float(expected.mean()))
        np.testing.assert_allclose(vp,expected.mean(axis=1))
        np.testing.assert_allclose(vn,expected.mean(axis=0))

    def test_partitions_are_not_reshuffled_and_chunks_cover_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for i,name in enumerate(['train','val','test']):
                frame=pd.DataFrame(np.zeros((20,800)),columns=PARTICLE_COLUMNS)
                frame['E_0']=100*(i+1)
                frame['is_signal_new']=[0,1]*10
                frame.to_hdf(root/f'{name}.h5',key='table',format='fixed' if name=='val' else 'table')
            with patch('prepare_official.verify'), patch('builtins.print'):
                prepare(root,train_count=12,val_count=6,test_count=20,chunk_size=7)
            fitting=loadmat(root/'official/fitting.mat')
            np.testing.assert_array_equal(fitting['partition'].ravel(),[1]*12+[2]*6)
            np.testing.assert_array_equal(fitting['particleData'][:,0],[100]*12+[200]*6)
            manifest=json.loads((root/'official/manifest.json').read_text())
            self.assertTrue(manifest['full_official_test'])
            rows=[]
            for chunk in manifest['test_chunks']:
                data=loadmat(root/'official'/chunk['file'])
                rows.extend(data['sourceRows'].ravel())
                self.assertTrue(np.all(data['particleData'][:,0]==300))
            self.assertEqual(rows,list(range(20)))
            with self.assertRaises(ValueError):
                read_rows(root/'train.h5',0,21)


if __name__=='__main__':
    unittest.main()
