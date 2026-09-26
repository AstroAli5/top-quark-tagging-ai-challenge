import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from convert_dataset import PARTICLE_COLUMNS
from prepare_official import sha256
from prepare_explanation_sample import prepare_sample


class ExplanationSample(unittest.TestCase):
    def test_prefix_uses_saved_boundaries_without_fitting_or_reassigned_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(np.zeros((20, 800)), columns=PARTICLE_COLUMNS)
            frame['E_0'] = np.arange(20)+1
            frame['is_signal_new'] = [0, 1]*10
            source = root/'test.h5'
            frame.to_hdf(source, key='table', format='table')
            chunks = [{'file': f'test_{start:06d}.mat', 'start': start, 'stop': min(start+7,20)}
                      for start in range(0,20,7)]
            manifest = {'test_count': 20, 'train_count': 100000, 'test_chunks': chunks,
                        'sources': {'test': {'sha256': sha256(source)}}}
            metadata = root/'metadata.json'
            metadata.write_text(json.dumps({'manifest': manifest}))
            with patch('prepare_explanation_sample.verify'), patch('builtins.print'):
                prepare_sample(metadata,source,root/'sample',max_jets=10)
            actual = json.loads((root/'sample/manifest.json').read_text())
            self.assertEqual(actual['materialized_test_rows'],14)
            self.assertEqual(actual['train_count'],100000)
            self.assertFalse((root/'sample/fitting.mat').exists())
            self.assertFalse((root/'sample/test_000014.mat').exists())
            first = loadmat(root/'sample/test_000000.mat')
            np.testing.assert_array_equal(first['sourceRows'].ravel(),np.arange(7))
            np.testing.assert_array_equal(first['particleData'][:,0],np.arange(7)+1)
            manifest['sources']['test']['sha256'] = 'changed'
            metadata.write_text(json.dumps({'manifest': manifest}))
            with patch('prepare_explanation_sample.verify'), self.assertRaisesRegex(ValueError,'differs'):
                prepare_sample(metadata,source,root/'changed',max_jets=10)
            self.assertFalse((root/'changed').exists())
