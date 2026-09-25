"""Recreate only the test chunks needed to explain a saved official study."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from download_dataset import download, verify
from prepare_official import read_rows, sha256


def prepare_sample(metadata_path, raw_test, output, max_jets=10000):
    metadata = json.loads(Path(metadata_path).read_text())
    manifest = copy.deepcopy(metadata['manifest'])
    raw_test, output = Path(raw_test), Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh explanation-data directory')
    if not 0 < max_jets <= manifest['test_count']:
        raise ValueError('Explanation sample must fit the saved test partition')
    verify(raw_test, 'test')
    if sha256(raw_test) != manifest['sources']['test']['sha256']:
        raise ValueError('Test source differs from the saved study')
    output.mkdir(parents=True)
    position = 0
    for chunk in manifest['test_chunks']:
        if position >= max_jets:
            break
        name = chunk['file']
        if Path(name).name != name or chunk['start'] != position or chunk['stop'] <= position:
            raise ValueError('Invalid saved chunk boundaries')
        x, y = read_rows(raw_test, chunk['start'], chunk['stop'])
        destination = output/name
        savemat(destination, {'particleData': x, 'labels': y,
                'sourceRows': np.arange(chunk['start'], chunk['stop'], dtype=np.int32).reshape(-1, 1)},
                do_compression=True)
        chunk['sha256'] = sha256(destination)
        position = chunk['stop']
    if position < max_jets:
        raise ValueError('Saved test chunks do not cover the explanation sample')
    # Retain the saved study's full boundaries for compatibility verification.
    # Only the stated prefix is materialized; this is not fitting/evaluation data.
    manifest['preparation_scope'] = 'explanation-only test prefix; no fitting.mat'
    manifest['materialized_test_rows'] = position
    manifest['requested_explanation_rows'] = max_jets
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Prepared {position} test rows for a {max_jets}-row explanation sample; no training data prepared.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--raw-test', type=Path, default=Path('data/test.h5'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-jets', type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh explanation-data directory')
    download(args.raw_test, 'test')
    prepare_sample(args.metadata, args.raw_test, args.output, args.max_jets)
