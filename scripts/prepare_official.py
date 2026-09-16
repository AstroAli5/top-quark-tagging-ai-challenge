"""Build disjoint fitting data and chunked official test data, with provenance."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
from convert_dataset import PARTICLE_COLUMNS
from download_dataset import download, verify


def read_rows(path, start, stop):
    frame = pd.read_hdf(path, key='table', start=start, stop=stop)
    particles = frame.loc[:, PARTICLE_COLUMNS].to_numpy(dtype=np.float32)
    labels = frame['is_signal_new'].to_numpy(dtype=np.float32).reshape(-1, 1)
    if not np.isfinite(particles).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid source data')
    if (particles[:, 0::4] < 0).any() or len(frame) != stop-start:
        raise ValueError('Unexpected row count or negative particle energy')
    return particles, labels


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(directory, train_count=50000, val_count=10000, test_count=None, chunk_size=5000):
    directory = Path(directory)
    destination = directory / 'official'
    destination.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ['train','val','test']:
        path = directory / f'{name}.h5'
        verify(path, name)
        with pd.HDFStore(path, mode='r') as store:
            storer = store.get_storer('table')
            available = getattr(storer, 'nrows', None)
            if available is None:
                available = storer.shape[0]
        sources[name] = {'file':path.name,'sha256':sha256(path),'available_rows':int(available)}
    if test_count is None:
        test_count = sources['test']['available_rows']
    if not (0 < train_count <= sources['train']['available_rows'] and
            0 < val_count <= sources['val']['available_rows'] and
            0 < test_count <= sources['test']['available_rows'] and chunk_size > 0):
        raise ValueError('Requested counts must fit their official source partitions')
    manifest = {'dataset':'10.5281/zenodo.2603256','license':'CC-BY-4.0',
                'selection':'First rows in source order; no cross-partition reassignment',
                'sources':sources,'train_count':train_count,'val_count':val_count,
                'test_count':test_count,'full_official_test':test_count==sources['test']['available_rows'],
                'test_chunks':[]}
    train_x, train_y = read_rows(directory/'train.h5',0,train_count)
    val_x, val_y = read_rows(directory/'val.h5',0,val_count)
    savemat(destination/'fitting.mat',{'particleData':np.concatenate([train_x,val_x]),
            'labels':np.concatenate([train_y,val_y]),
            'partition':np.r_[np.ones(train_count),np.full(val_count,2)].astype(np.uint8).reshape(-1,1),
            'provenance_json':json.dumps(manifest)},do_compression=True)
    for start in range(0,test_count,chunk_size):
        stop = min(start+chunk_size,test_count)
        x,y = read_rows(directory/'test.h5',start,stop)
        name = f'test_{start:06d}.mat'
        savemat(destination/name,{'particleData':x,'labels':y,
                'sourceRows':np.arange(start,stop,dtype=np.int32).reshape(-1,1)},do_compression=True)
        manifest['test_chunks'].append({'file':name,'start':start,'stop':stop,'sha256':sha256(destination/name)})
    manifest['fitting_sha256'] = sha256(destination/'fitting.mat')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda p: download(args.data_dir/f'{p}.h5',p),['train','val','test']))
    prepare(args.data_dir)


if __name__=='__main__':
    main()
