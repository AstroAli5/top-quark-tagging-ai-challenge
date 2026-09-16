"""Download and verify official partitions from Zenodo record 2603256."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

URL = "https://zenodo.org/api/records/2603256/files/train.h5/content"
EXPECTED_BYTES = 1038496555
EXPECTED_MD5 = "45663819f47c13724f67eb0fd80bfa5c"
PARTITIONS = {
    'train': (EXPECTED_BYTES, EXPECTED_MD5),
    'val': (347378076, 'dca4b7248027618f041f9baa86d360fc'),
    'test': (347849376, '13163479dee30a5fe546e4536cc3d04d'),
}


def verify(path, partition='train'):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    expected_bytes, expected_md5 = PARTITIONS[partition]
    if Path(path).stat().st_size != expected_bytes or digest.hexdigest() != expected_md5:
        raise ValueError("Dataset size/checksum does not match the published training file")


def download(destination, partition='train'):
    destination = Path(destination)
    if destination.exists():
        verify(destination, partition)
        print(f"Verified existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep partial downloads separate from the trusted final filename.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            url = f'https://zenodo.org/api/records/2603256/files/{partition}.h5/content'
            expected_bytes, _ = PARTITIONS[partition]
            request = urllib.request.Request(url, headers={"User-Agent": "top-quark-research/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                size = 0
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    size += len(block)
                    if size > expected_bytes:
                        raise ValueError("Download exceeds the published file size")
                    out.write(block)
        verify(temporary, partition)
        temporary.replace(destination)
        print(f"Downloaded and verified {destination} ({expected_bytes:,} bytes)")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', choices=PARTITIONS, default='train')
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        download(args.output or Path('data') / f'{args.partition}.h5', args.partition)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")


if __name__ == "__main__":
    main()
