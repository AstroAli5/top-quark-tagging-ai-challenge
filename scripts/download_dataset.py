"""Download and verify the official training file from Zenodo record 2603256."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

URL = "https://zenodo.org/api/records/2603256/files/train.h5/content"
EXPECTED_BYTES = 1038496555
EXPECTED_MD5 = "45663819f47c13724f67eb0fd80bfa5c"


def verify(path):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if Path(path).stat().st_size != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_MD5:
        raise ValueError("Dataset size/checksum does not match the published training file")


def download(destination):
    destination = Path(destination)
    if destination.exists():
        verify(destination)
        print(f"Verified existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep partial downloads separate from the trusted final filename.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            request = urllib.request.Request(URL, headers={"User-Agent": "top-quark-research/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                size = 0
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    size += len(block)
                    if size > EXPECTED_BYTES:
                        raise ValueError("Download exceeds the published file size")
                    out.write(block)
        verify(temporary)
        temporary.replace(destination)
        print(f"Downloaded and verified {destination} ({EXPECTED_BYTES:,} bytes)")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/train.h5"))
    args = parser.parse_args()
    try:
        download(args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")


if __name__ == "__main__":
    main()
