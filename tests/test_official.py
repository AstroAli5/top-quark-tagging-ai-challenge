import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.io import loadmat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_official import prepare, read_rows
from convert_dataset import PARTICLE_COLUMNS
from summarize_experiment import auc_influences


class OfficialPartitions(unittest.TestCase):
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
