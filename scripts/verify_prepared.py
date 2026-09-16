"""Verify prepared-file hashes before evaluating restored checkpoints."""
import hashlib
import json
from pathlib import Path


def verify(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    files=[('fitting.mat',manifest['fitting_sha256'])]
    files += [(chunk['file'],chunk['sha256']) for chunk in manifest['test_chunks']]
    for name,expected in files:
        with (directory/name).open('rb') as handle:
            actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=expected:
            raise ValueError(f'Prepared file checksum mismatch: {name}')
    print(f'Verified {len(files)} prepared files; {manifest["test_count"]} test rows.')


if __name__=='__main__':
    verify('data/official')
