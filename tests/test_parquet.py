"""Check bounded conversion, exact row/label identity and cache integrity."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_parquet
from convert_dataset import PARTICLE_COLUMNS

class TestParquet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'train.h5'
        values = np.arange(11*800,dtype=np.float32).reshape(11,800)/100
        self.frame = pd.DataFrame(values,columns=PARTICLE_COLUMNS)
        self.frame['is_signal_new'] = np.arange(11)%2
        self.frame.to_hdf(self.source,key='table',format='table')

    def testBoundedRoundTripAndCacheIdentity(self):
        with patch.object(prepare_parquet.pd,'read_hdf',wraps=pd.read_hdf) as reader:
            m = prepare_parquet.convert_partition(self.source,self.root/'output','train',9,4,False)
        self.assertEqual([(c.kwargs['start'],c.kwargs['stop']) for c in reader.call_args_list],[(0,4),(4,8),(8,9)])
        result = pd.concat([pd.read_parquet(self.root/'output'/c['file']) for c in m['chunks']],ignore_index=True)
        np.testing.assert_array_equal(result[PARTICLE_COLUMNS],self.frame[PARTICLE_COLUMNS].iloc[:9])
        np.testing.assert_array_equal(result.SourceRow,np.arange(9))
        np.testing.assert_array_equal(result.Label,self.frame.is_signal_new.iloc[:9])
        self.assertFalse(m['verified_official_source'])
        self.assertEqual(m,prepare_parquet.convert_partition(self.source,self.root/'output','train',9,4,False))
        with self.assertRaisesRegex(ValueError,'differs'):
            prepare_parquet.convert_partition(self.source,self.root/'output','train',10,4,False)
        (self.root/'output'/m['chunks'][0]['file']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'checksum'):
            prepare_parquet.convert_partition(self.source,self.root/'output','train',9,4,False)

    def testAllRowsAndInvalidCounts(self):
        m = prepare_parquet.convert_partition(self.source,self.root/'all','train',None,4,False)
        self.assertEqual(m['selected_rows'],11)
        self.assertEqual(m['class_counts'],[6,5])
        for count in [0,-1,1.5,12]:
            with self.assertRaises(ValueError):
                prepare_parquet.convert_partition(self.source,self.root/'bad','train',count,4,False)
        self.assertFalse((self.root/'bad').exists())

    def testInvalidLabelsAndIncompleteOutput(self):
        self.frame.loc[2,'is_signal_new'] = 2
        self.frame.to_hdf(self.source,key='table',format='table',mode='w')
        with self.assertRaisesRegex(ValueError,'labels'):
            prepare_parquet.convert_partition(self.source,self.root/'bad','train',5,4,False)
        with self.assertRaisesRegex(FileExistsError,'Incomplete'):
            prepare_parquet.convert_partition(self.source,self.root/'bad','train',5,4,False)
