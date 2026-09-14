"""Convert the published pandas jet table to MATLAB's N x 800 schema."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import savemat

PARTICLE_COLUMNS = [
    f"{component}_{particle}"
    for particle in range(200)
    for component in ("E", "PX", "PY", "PZ")
]


def convert_dataset(source, destination, max_jets=50000, key="/table", force=False):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if destination.suffix.lower() != ".mat":
        raise ValueError("The output filename must end in .mat")
    if max_jets < 1:
        raise ValueError("max_jets must be positive")
    if destination.exists() and not force:
        raise FileExistsError(f"Output exists: {destination}. Use --force to replace it.")

    frame = pd.read_hdf(source, key=key, start=0, stop=max_jets)
    required = PARTICLE_COLUMNS + ["is_signal_new"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing[:8])}")
    if frame.empty:
        raise ValueError("The selected dataset is empty")
    if not frame.columns.is_unique:
        raise ValueError("Dataset column names must be unique")
    particles = frame.loc[:, PARTICLE_COLUMNS].to_numpy(dtype=np.float32)
    labels = frame["is_signal_new"].to_numpy()
    if not np.isfinite(particles).all():
        raise ValueError("Particle data contains non-finite values")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Labels must contain both classes 0 and 1")
    if (particles[:, 0::4] < 0).any():
        raise ValueError("Particle energies must be nonnegative")

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    provenance = {
        "source_file": source.name,
        "source_sha256": digest.hexdigest(),
        "hdf_key": key,
        "selection": "first rows in source order",
        "requested_max_jets": max_jets,
        "converted_jets": len(frame),
        "column_order": "(E, PX, PY, PZ) for particles 0..199",
        "signal_jets": int(np.sum(labels == 1)),
        "background_jets": int(np.sum(labels == 0)),
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        str(destination),
        {
            "particleData": particles,
            "labels": labels.astype(np.float32).reshape(-1, 1),
            "provenance_json": json.dumps(provenance),
        },
        appendmat=False,
        do_compression=True,
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/jets_real_50k.mat"))
    parser.add_argument("--max-jets", type=int, default=50000)
    parser.add_argument("--key", default="/table")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = convert_dataset(args.input, args.output, args.max_jets, args.key, args.force)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")
    print(f"Saved {result['converted_jets']} jets to {args.output}")
    print(f"Source SHA-256: {result['source_sha256']}")


if __name__ == "__main__":
    main()
