"""Per-campaign STORAGE_ROOT layout + server-driven cleanup. No network."""
from __future__ import annotations

import os
import time
from collections import namedtuple
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.jobs.download import DownloadJob
from app.jobs.transcribe import TranscribeJob
from app.models.job import Job
from app.services.api_client import APIClient, MockAPIClient
from app.services.retention import sweep_media
from app.services.storage_cleanup import run_cleanup, scan
from app.utils.clip_storage import ClipStorage, ClipStorageError
from app.utils.storage_layout import StorageLayout, StoragePathError, is_within, safe_remove

A1 = "11111111-1111-1111-1111-111111111111"
A2 = "22222222-2222-2222-2222-222222222222"
A3 = "33333333-3333-3333-3333-333333333333"
Usage = namedtuple("Usage", "total used free")


def _settings(tmp_path: Path, **kw) -> Settings:
    base = dict(
        working_directory=str(tmp_path / "work"),
        clip_storage_root=str(tmp_path / "legacy_clips"),
        storage_root=str(tmp_path / "root"),
        media_retention_enabled=True,
        media_retention_hours=24,
    )
    base.update(kw)
    s = Settings(**base)
    if s.storage_root:
        Path(s.storage_root).mkdir(parents=True, exist_ok=True)
    return s


def _write(p: Path, data: bytes = b"x" * 10) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _age(p: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(p, (t, t))


class FakeAPI:
    def __init__(self, answer=None, exc=None):
        self.answer, self.exc, self.bodies = answer or {}, exc, []

    def storage_check(self, body):
        self.bodies.append(body)
        if self.exc:
            raise self.exc
        return self.answer


def _no_pressure(_):
    return Usage(100, 10, 90)


# ---------------------------------------------------------------- config/layout
def test_storage_root_empty_means_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_ROOT", "")
    s = Settings(working_directory=str(tmp_path))
    assert s.storage_root is None
    assert s.storage_cleanup_enabled is False and s.storage_cleanup_dry_run is True
    assert s.clip_retention_days == 7 and s.disk_high_watermark_pct == 80
    monkeypatch.setenv("STORAGE_ROOT", "/srv/datos/clipping")
    assert Settings(working_directory=str(tmp_path)).storage_root == Path("/srv/datos/clipping")


def test_layout_paths(tmp_path):
    lay = StorageLayout(tmp_path)
    assert lay.sources_dir(7, A1) == tmp_path / "7" / "sources" / A1
    assert lay.work_dir("7", A1) == tmp_path / "7" / "work" / A1
    assert lay.clips_dir(7, "pending_upload") == tmp_path / "7" / "clips" / "pending"
    assert lay.clips_dir(7, "uploaded") == tmp_path / "7" / "clips" / "done"


@pytest.mark.parametrize("cid", ["..", "../7", "7/..", "/etc", "a", "", "7 8"])
def test_layout_rejects_bad_campaign(tmp_path, cid):
    with pytest.raises(StoragePathError):
        StorageLayout(tmp_path).campaign_dir(cid)


@pytest.mark.parametrize("aid", ["..", "../x", "a/b", "a\\b", ".hidden", ""])
def test_layout_rejects_bad_asset(tmp_path, aid):
    with pytest.raises(StoragePathError):
        StorageLayout(tmp_path).sources_dir(7, aid)


# ------------------------------------------------------------------ safe_remove
def test_safe_remove_refuses_outside_and_root(tmp_path):
    root = tmp_path / "root"
    outside = _write(tmp_path / "outside" / "f.bin")
    root.mkdir()
    assert safe_remove(outside, root, dry_run=False, reason="t") == 0
    assert safe_remove(root / ".." / "outside" / "f.bin", root, dry_run=False, reason="t") == 0
    assert safe_remove(root, root, dry_run=False, reason="t") == 0
    assert outside.exists() and root.exists()


def test_safe_remove_refuses_symlink_and_symlinked_parent(tmp_path):
    root = tmp_path / "root"
    victim = _write(tmp_path / "victim" / "keep.bin")
    (root / "7").mkdir(parents=True)
    link = root / "7" / "sources"
    link.symlink_to(victim.parent, target_is_directory=True)
    assert safe_remove(link, root, dry_run=False, reason="t") == 0
    # path through a symlinked dir resolves outside the root
    assert safe_remove(link / "keep.bin", root, dry_run=False, reason="t") == 0
    assert victim.exists() and link.is_symlink()


def test_safe_remove_tree_does_not_follow_inner_symlink(tmp_path):
    root = tmp_path / "root"
    victim = _write(tmp_path / "victim" / "keep.bin")
    tree = root / "7" / "sources" / A1
    _write(tree / "source.mp4")
    (tree / "evil").symlink_to(victim.parent, target_is_directory=True)
    freed = safe_remove(tree, root, dry_run=False, reason="t")
    assert freed == 10
    assert not tree.exists() and victim.exists()


def test_safe_remove_dry_run_keeps(tmp_path):
    root = tmp_path / "root"
    f = _write(root / "7" / "sources" / A1 / "source.mp4")
    assert safe_remove(f.parent, root, dry_run=True, reason="t") == 10
    assert f.exists()
    assert is_within(f, root) and not is_within(root, root)


# --------------------------------------------------------------- job wiring
def test_clip_storage_campaign_layout(tmp_path):
    cs = ClipStorage(tmp_path, 7, campaign_layout=True)
    assert cs.path_for("c1", "pending_upload") == tmp_path / "7" / "clips" / "pending" / "c1.mp4"
    assert cs.path_for("c1", "archived") == tmp_path / "7" / "clips" / "done" / "c1.mp4"
    legacy = ClipStorage(tmp_path, 7)
    assert legacy.path_for("c1", "pending_upload") == tmp_path / "7" / "pending_upload" / "c1.mp4"
    with pytest.raises(ClipStorageError):
        ClipStorage(tmp_path, "../x", campaign_layout=True)


def test_clip_storage_for_settings(tmp_path):
    s = _settings(tmp_path)
    assert ClipStorage.for_settings(s, 7).path_for("c", "uploaded").parent == Path(s.storage_root) / "7" / "clips" / "done"
    s2 = _settings(tmp_path, storage_root="")
    assert ClipStorage.for_settings(s2, 7).path_for("c", "uploaded").parent == Path(s2.clip_storage_root) / "7" / "uploaded"


def test_download_persist_campaign_layout(tmp_path):
    s = _settings(tmp_path)
    s.ensure_directories()
    src = _write(tmp_path / "dl.mp4")
    job = DownloadJob(s, Job(id="j1", type="download", payload={}))
    dest = job._persist(src, {"asset_id": A1, "campaign_id": 7, "filename": "x.mp4"})
    assert dest == Path(s.storage_root) / "7" / "sources" / A1 / "source.mp4" and dest.exists()
    # no campaign_id (old payload) -> legacy path
    dest2 = job._persist(src, {"asset_id": A2, "filename": "x.mp4"})
    assert dest2 == s.downloads_dir / A2 / "source.mp4"
    assert job._persist(src, {"asset_id": A3, "campaign_id": "../9", "filename": "x.mp4"}) == s.downloads_dir / A3 / "source.mp4"
    # legacy settings
    s2 = _settings(tmp_path, storage_root="")
    job2 = DownloadJob(s2, Job(id="j2", type="download", payload={}))
    assert job2._persist(src, {"asset_id": A3, "campaign_id": 7, "filename": "x.mp4"}) == s2.downloads_dir / A3 / "source.mp4"


def test_transcribe_work_dir(tmp_path):
    s = _settings(tmp_path)
    s.ensure_directories()
    job = TranscribeJob(s, Job(id="t1", type="transcribe", payload={}))
    assert job._work_dir({"asset_id": A1, "campaign_id": 7}) == Path(s.storage_root) / "7" / "work" / A1
    assert job._work_dir({"asset_id": A1}) is None
    assert job._work_dir({"asset_id": "../x", "campaign_id": 7}) is None
    s2 = _settings(tmp_path, storage_root="")
    assert TranscribeJob(s2, Job(id="t2", type="transcribe", payload={}))._work_dir({"asset_id": A1, "campaign_id": 7}) is None


# ------------------------------------------------------------------ retention
def test_retention_skips_storage_root(tmp_path):
    s = _settings(tmp_path, clip_storage_root=str(tmp_path / "root"))
    f = _write(Path(s.storage_root) / "7" / "clips" / "pending" / "c.mp4")
    _age(f, 30 * 86400)
    assert sweep_media(s)["removed"] == 0 and f.exists()


# ------------------------------------------------------------------- cleanup
def _tree(s: Settings) -> dict[str, Path]:
    r = Path(s.storage_root)
    return {
        "src1": _write(r / "7" / "sources" / A1 / "source.mp4"),
        "audio1": _write(r / "7" / "work" / A1 / "audio.wav"),
        "tr1": _write(r / "7" / "work" / A1 / "transcript.json", b"{}"),
        "src2": _write(r / "7" / "sources" / A2 / "source.mp4"),
        "src9": _write(r / "9" / "sources" / A3 / "source.mp4"),
        "clip_rej": _write(r / "7" / "clips" / "pending" / "c-rej.mp4"),
        "clip_new": _write(r / "7" / "clips" / "pending" / "c-new.mp4"),
        "clip_old": _write(r / "7" / "clips" / "done" / "c-old.mp4"),
        "clip_old_active": _write(r / "7" / "clips" / "pending" / "c-act.mp4"),
    }


ANSWER = {
    "campaigns": {"7": {"known": True, "status": "active", "delete": False},
                  "9": {"known": True, "status": "archived", "delete": True}},
    "assets": {A1: {"known": True, "campaign_id": 7, "done": True, "idle": True},
               A2: {"known": True, "campaign_id": 7, "done": False, "idle": False},
               A3: {"known": True, "campaign_id": 9, "done": False, "idle": True}},
    "clips": {"c-rej": {"known": True, "status": "rejected", "active": False, "release": True},
              "c-new": {"known": True, "status": "approved", "active": False, "release": False},
              "c-old": {"known": True, "status": "approved", "active": False, "release": False},
              "c-act": {"known": True, "status": "approved", "active": True, "release": False}},
}


def test_cleanup_apply(tmp_path):
    s = _settings(tmp_path)
    t = _tree(s)
    for k in ("clip_old", "clip_old_active"):
        _age(t[k], 8 * 86400)
    api = FakeAPI(ANSWER)
    out = run_cleanup(s, api, dry_run=False, disk_usage=_no_pressure)
    body = api.bodies[0]
    assert body["campaign_ids"] == ["7", "9"] and set(body["asset_ids"]) == {A1, A2, A3}
    assert set(body["clip_ids"]) == {"c-rej", "c-new", "c-old", "c-act"}
    assert not (Path(s.storage_root) / "9").exists()          # archived campaign
    assert not t["src1"].exists() and not t["audio1"].exists()  # decided asset
    assert t["tr1"].exists()                                    # transcript kept
    assert t["src2"].exists()                                   # active jobs -> kept
    assert not t["clip_rej"].exists()                           # rejected clip
    assert t["clip_new"].exists()                               # fresh, not decided
    assert not t["clip_old"].exists()                           # > 7 days
    assert t["clip_old_active"].exists()                        # job uses it
    assert out["deleted"] == {"campaign_archived": 1, "asset_decided": 1, "clip_rejected": 1, "clip_ttl": 1}


def test_cleanup_dry_run_deletes_nothing(tmp_path):
    s = _settings(tmp_path)
    t = _tree(s)
    _age(t["clip_old"], 8 * 86400)
    out = run_cleanup(s, FakeAPI(ANSWER), disk_usage=_no_pressure)  # settings default: dry-run
    assert out["dry_run"] is True and out["deleted"]
    assert all(p.exists() for p in t.values())


def test_cleanup_server_unreachable_deletes_nothing(tmp_path):
    s = _settings(tmp_path)
    t = _tree(s)
    out = run_cleanup(s, FakeAPI(exc=httpx.ConnectError("down")), dry_run=False, disk_usage=_no_pressure)
    assert out["skipped"] == "server_check_failed"
    assert all(p.exists() for p in t.values())


def test_cleanup_unknown_ids_never_deleted(tmp_path):
    s = _settings(tmp_path)
    t = _tree(s)
    out = run_cleanup(s, FakeAPI({}), dry_run=False, disk_usage=lambda _: Usage(100, 99, 1))
    assert all(p.exists() for p in t.values())
    assert out["deleted"] == {}


def test_cleanup_asset_in_other_campaign_left_alone(tmp_path):
    s = _settings(tmp_path)
    t = _tree(s)
    ans = {"assets": {A1: {"known": True, "campaign_id": 8, "done": True, "idle": True}}}
    run_cleanup(s, FakeAPI(ans), dry_run=False, disk_usage=_no_pressure)
    assert t["src1"].exists()


def test_cleanup_disk_pressure_oldest_processed_first(tmp_path):
    s = _settings(tmp_path)
    r = Path(s.storage_root)
    old = _write(r / "7" / "sources" / A1 / "source.mp4", b"x" * 30)
    _write(r / "7" / "work" / A1 / "transcript.json", b"{}")
    new = _write(r / "7" / "sources" / A2 / "source.mp4", b"x" * 30)
    _write(r / "7" / "work" / A2 / "transcript.json", b"{}")
    unprocessed = _write(r / "7" / "sources" / A3 / "source.mp4", b"x" * 30)
    busy = "44444444-4444-4444-4444-444444444444"
    busy_src = _write(r / "7" / "sources" / busy / "source.mp4", b"x" * 30)
    _write(r / "7" / "work" / busy / "transcript.json", b"{}")
    for p, age in ((old, 3000), (new, 100), (unprocessed, 9000), (busy_src, 9999)):
        _age(p, age)
    idle = {"known": True, "campaign_id": 7, "done": False, "idle": True}
    ans = {"assets": {A1: idle, A2: idle, A3: idle,
                      busy: {"known": True, "campaign_id": 7, "done": False, "idle": False}}}
    # 165/200 = 82.5 % -> freeing the oldest 30 bytes gives 67.5 % -> stop
    out = run_cleanup(s, FakeAPI(ans), dry_run=False, disk_usage=lambda _: Usage(200, 165, 35))
    assert not old.exists()                 # oldest processed goes first
    assert new.exists()                     # stopped once below 80 %
    assert unprocessed.exists()             # no transcript yet
    assert busy_src.exists()                # jobs pending/running
    assert (r / "7" / "work" / A1 / "transcript.json").exists()
    assert out["deleted"] == {"disk_pressure": 1}


def test_scan_ignores_bad_names_and_symlinks(tmp_path):
    root = tmp_path / "root"
    _write(root / "notacampaign" / "sources" / A1 / "s.mp4")
    _write(root / "7" / "sources" / ".." / "x")
    outside = tmp_path / "out"
    _write(outside / A2 / "s.mp4")
    (root / "8").mkdir()
    (root / "8" / "sources").symlink_to(outside, target_is_directory=True)
    inv = scan(root)
    assert set(inv["campaigns"]) == {"7", "8"}
    assert inv["assets"] == {}


def test_cleanup_without_storage_root_is_noop(tmp_path):
    s = _settings(tmp_path, storage_root="")
    assert run_cleanup(s, FakeAPI(ANSWER), dry_run=False)["skipped"] == "no_storage_root"


# ----------------------------------------------------------------- API client
def test_api_client_storage_check_posts(tmp_path):
    s = Settings(worker_id="w", api_base_url="http://vps.test", api_token="t", working_directory=str(tmp_path))
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, request.read()
        return httpx.Response(200, json={"assets": {}, "campaigns": {}, "clips": {}})

    api = APIClient(s)
    api.client = httpx.Client(base_url="http://vps.test", transport=httpx.MockTransport(handler))
    assert api.storage_check({"asset_ids": [A1]}) == {"assets": {}, "campaigns": {}, "clips": {}}
    assert seen["path"] == "/worker/storage/check" and A1.encode() in seen["body"]
    assert MockAPIClient(s).storage_check({}) == {"assets": {}, "campaigns": {}, "clips": {}}


# ------------------------------------------------------------------ migration
def test_migration_script_dry_run_then_apply(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "migrate_storage_layout", Path(__file__).resolve().parents[1] / "scripts" / "migrate_storage_layout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    legacy, root = tmp_path / "clips", tmp_path / "root"
    p = _write(legacy / "7" / "pending_upload" / "c1.mp4")
    u = _write(legacy / "7" / "uploaded" / "c2.mp4")
    _write(legacy / "bad" / "pending_upload" / "c3.mp4")
    assert mod.main(["--from", str(legacy), "--to", str(root)]) == 0
    assert p.exists() and not root.exists()
    assert "would move=2" in capsys.readouterr().out
    assert mod.main(["--from", str(legacy), "--to", str(root), "--apply"]) == 0
    assert (root / "7" / "clips" / "pending" / "c1.mp4").exists()
    assert (root / "7" / "clips" / "done" / "c2.mp4").exists()
    assert not p.exists() and not u.exists()
