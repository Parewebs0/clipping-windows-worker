"""Refresh must request exactly youtube.upload + youtube.readonly (invalid_scope on 18 Oct)."""

from __future__ import annotations

from unittest.mock import MagicMock

import app.services.youtube_upload as yu

EXPECTED = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


def _run(tmp_path, monkeypatch, privacy="public"):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"x")
    yt = MagicMock()
    req = MagicMock()
    req.next_chunk.return_value = (None, {"id": "vid1"})
    yt.videos.return_value.insert.return_value = req
    creds_cls = MagicMock()
    monkeypatch.setattr(yu, "build", lambda *a, **k: yt)
    monkeypatch.setattr(yu, "Credentials", creds_cls)
    monkeypatch.setattr(yu, "MediaFileUpload", MagicMock())
    out = yu.upload_short(file_path=f, title="t", description="d", client_id="a", client_secret="b",
                          refresh_token="c", privacy=privacy)
    return out, creds_cls, yt


def test_refresh_requests_exact_scopes(tmp_path, monkeypatch):
    out, creds_cls, _ = _run(tmp_path, monkeypatch)
    assert out["video_id"] == "vid1"
    assert creds_cls.call_args.kwargs["scopes"] == EXPECTED
    creds_cls.return_value.refresh.assert_called_once()


def test_full_youtube_scope_never_requested():
    assert "https://www.googleapis.com/auth/youtube" not in yu.SCOPES
    assert list(yu.SCOPES) == EXPECTED


def test_google_auth_refresh_grant_gets_exact_scopes(monkeypatch):
    """Real google-auth Credentials: refresh_grant receives exactly those scopes."""
    from google.oauth2 import reauth
    from google.oauth2.credentials import Credentials

    seen = {}

    def fake_refresh_grant(request, token_uri, refresh_token, client_id, client_secret, scopes=None, rapt_token=None, **kw):
        seen["scopes"] = list(scopes or [])
        return "tok", refresh_token, None, {"scope": " ".join(EXPECTED)}, None

    monkeypatch.setattr(reauth, "refresh_grant", fake_refresh_grant)
    creds = Credentials(token=None, refresh_token="r", token_uri=yu.TOKEN_URI, client_id="a",
                        client_secret="b", scopes=list(yu.SCOPES))
    creds.refresh(MagicMock())
    assert seen["scopes"] == EXPECTED


def test_public_privacy_passed(tmp_path, monkeypatch):
    _, _, yt = _run(tmp_path, monkeypatch)
    body = yt.videos.return_value.insert.call_args.kwargs["body"]
    assert body["status"]["privacyStatus"] == "public"
