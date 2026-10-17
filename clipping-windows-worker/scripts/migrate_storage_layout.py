"""Move legacy clips into the STORAGE_ROOT layout. Dry-run unless --apply.

    <clip_storage_root>/<cid>/pending_upload/<clip>.mp4 -> <root>/<cid>/clips/pending/
    <clip_storage_root>/<cid>/{uploaded,archived}/<clip>.mp4 -> <root>/<cid>/clips/done/

Only clips move. Sources under <working_directory>/downloads/ stay where they
are: asset.local_path in the DB points at them and queued render jobs read that
absolute path. Legacy retention (MEDIA_RETENTION_HOURS) removes them later.
Publish finds a moved pending clip through ClipStorage.path_for, so a stale
file_path in a queued publish job is fine.

    python scripts/migrate_storage_layout.py --from /data/clips --to /srv/datos/clipping
    python scripts/migrate_storage_layout.py --from /data/clips --to /srv/datos/clipping --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.storage_layout import CLIP_DIRS, StoragePathError, campaign_key  # noqa: E402


def plan(src_root: Path, dst_root: Path) -> list[tuple[Path, Path]]:
    moves = []
    if not src_root.is_dir():
        return moves
    for cdir in sorted(src_root.iterdir()):
        if not cdir.is_dir() or cdir.is_symlink():
            continue
        try:
            cid = campaign_key(cdir.name)
        except StoragePathError:
            continue
        for location, parts in CLIP_DIRS.items():
            ldir = cdir / location
            if not ldir.is_dir() or ldir.is_symlink():
                continue
            for f in sorted(ldir.glob("*.mp4")):
                if f.is_file() and not f.is_symlink():
                    moves.append((f, dst_root.joinpath(cid, *parts, f.name)))
    return moves


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="src", required=True, type=Path, help="legacy CLIP_STORAGE_ROOT")
    p.add_argument("--to", dest="dst", required=True, type=Path, help="STORAGE_ROOT")
    p.add_argument("--apply", action="store_true", help="really move (default: dry-run)")
    a = p.parse_args(argv)
    if a.src.resolve() == a.dst.resolve():
        print("--from and --to must differ", file=sys.stderr)
        return 2
    moved = skipped = 0
    for src, dst in plan(a.src, a.dst):
        if dst.exists():
            print(f"SKIP exists {dst}")
            skipped += 1
            continue
        print(f"{'MOVE' if a.apply else 'DRY-RUN move'} {src} -> {dst}")
        if a.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        moved += 1
    print(f"{'moved' if a.apply else 'would move'}={moved} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
