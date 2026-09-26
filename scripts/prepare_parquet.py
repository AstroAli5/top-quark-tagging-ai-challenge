"""Bounded HDF5 -> Parquet preparation, called from MATLAB via pyrun.

Official partitions remain separate. Every row has an explicit source ordinal;
no fitted preprocessing or random split occurs in Python.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow

from convert_dataset import PARTICLE_COLUMNS
from download_dataset import download, verify


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def convert_partition(source, destination, partition, count, chunk_rows=2000,
                      verify_source=True):
    source, destination = Path(source), Path(destination)
    if partition not in ('train', 'val', 'test'):
        raise ValueError('Unknown partition')
    if int(chunk_rows) != chunk_rows or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    if count is not None and (int(count) != count or count < 1):
        raise ValueError('count must be a positive integer or None for all rows')
    if verify_source:
        verify(source, partition)
    with pd.HDFStore(source, mode='r') as store:
        available = int(store.get_storer('table').nrows)
    selected = available if count is None else int(count)
    if selected > available:
        raise ValueError('Requested more rows than the source contains')
    identity = dict(partition=partition, source_sha256=sha256(source),
                    available_rows=available, selected_rows=selected,
                    chunk_rows=int(chunk_rows), verified_official_source=verify_source)
    manifest_path = destination / 'manifest.json'
    if destination.exists():
        if not manifest_path.is_file():
            raise FileExistsError('Incomplete output; use a new destination')
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError('Existing Parquet selection or source differs')
        for chunk in previous['chunks']:
            if sha256(destination / chunk['file']) != chunk['sha256']:
                raise ValueError('Existing Parquet checksum mismatch')
        return previous
    destination.mkdir(parents=True)
    started = time.perf_counter()
    chunks, class_counts = [], np.zeros(2, dtype=np.int64)
    for start in range(0, selected, int(chunk_rows)):
        stop = min(start + int(chunk_rows), selected)
        frame = pd.read_hdf(source, key='table', start=start, stop=stop)
        if len(frame) != stop-start or not frame.columns.is_unique:
            raise ValueError('Incomplete HDF5 read or duplicate columns')
        particles = frame.loc[:, PARTICLE_COLUMNS].to_numpy(dtype=np.float32)
        labels = frame['is_signal_new'].to_numpy()
        if not np.isfinite(particles).all() or (particles[:, 0::4] < 0).any():
            raise ValueError('Invalid particle values')
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('Invalid labels')
        output = pd.DataFrame(particles, columns=PARTICLE_COLUMNS)
        output['Label'] = labels.astype(np.int8)
        output['SourceRow'] = np.arange(start, stop, dtype=np.int64)
        filename = f'part_{start:09d}.parquet'
        output.to_parquet(destination / filename, engine='pyarrow',
                          compression='gzip', index=False, row_group_size=int(chunk_rows))
        chunks.append(dict(file=filename, start=start, stop=stop,
                           sha256=sha256(destination / filename)))
        class_counts += np.bincount(labels.astype(int), minlength=2)
    if not np.all(class_counts > 0):
        raise ValueError('The selected partition must contain both classes')
    manifest = dict(schema_version=1, **identity, chunks=chunks,
                    class_counts=class_counts.tolist(),
                    selection='first rows in official source order; partitions stay separate',
                    column_order=PARTICLE_COLUMNS,
                    pandas_version=pd.__version__, pyarrow_version=pyarrow.__version__,
                    conversion_seconds=time.perf_counter()-started)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def prepare(raw_dir, output_dir, counts, chunk_rows=2000, download_sources=True):
    """download_sources=False is for labelled synthetic integration fixtures."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    counts = json.loads(counts) if isinstance(counts, str) else counts
    if set(counts) != {'train', 'val', 'test'}:
        raise ValueError('Specify separate train, val and test counts')
    output_dir.mkdir(parents=True, exist_ok=True)
    partitions = {}
    for partition in ('train', 'val', 'test'):
        source = raw_dir / (partition+'.h5')
        if download_sources:
            download(source, partition)
        partitions[partition] = convert_partition(
            source, output_dir / partition, partition, counts[partition],
            chunk_rows, verify_source=download_sources)
    manifest = dict(schema_version=1, dataset='10.5281/zenodo.2603256',
                    verified_official_source=download_sources, partitions=partitions)
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
