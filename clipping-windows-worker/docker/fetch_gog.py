"""Download the gog CLI release tarball, verify its sha256 and install the binary.

Used only at image build time (see Dockerfile). Stdlib only, so the slim base
image does not need curl.

Usage: python fetch_gog.py <url> <sha256> <dest>
"""
import hashlib
import io
import sys
import tarfile
import urllib.request


def main() -> int:
    url, expected, dest = sys.argv[1], sys.argv[2].lower(), sys.argv[3]
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    if got != expected:
        print(f"gog checksum mismatch for {url}: expected {expected}, got {got}", file=sys.stderr)
        return 1
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        member = next(
            (m for m in tf.getmembers() if m.isfile() and m.name.rsplit("/", 1)[-1] == "gog"),
            None,
        )
        if member is None:
            print("gog binary not found in tarball", file=sys.stderr)
            return 1
        with open(dest, "wb") as out:
            out.write(tf.extractfile(member).read())
    print(f"installed gog from {url} -> {dest} (sha256 {got})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
