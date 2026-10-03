"""Issue #6: paid promotion flag on YouTube upload (paidProductPlacementDetails)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import app.services.youtube_upload as yu


@pytest.mark.parametrize("paid,expect_part", [(True, "snippet,status,paidProductPlacementDetails"), (False, "snippet,status")])
def test_paid_promotion_part_and_body(tmp_path, monkeypatch, paid, expect_part):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"x")
    calls = {}
    yt = MagicMock()

    def insert(part, body, media_body):
        calls["part"], calls["body"] = part, body
        req = MagicMock()
        req.next_chunk.return_value = (None, {"id": "vid1"})
        return req

    yt.videos.return_value.insert.side_effect = insert
    monkeypatch.setattr(yu, "build", lambda *a, **k: yt)
    monkeypatch.setattr(yu, "Credentials", MagicMock())
    monkeypatch.setattr(yu, "MediaFileUpload", MagicMock())
    out = yu.upload_short(file_path=f, title="t", description="d", client_id="a", client_secret="b",
                          refresh_token="c", paid_promotion=paid)
    assert out["video_id"] == "vid1"
    assert calls["part"] == expect_part
    assert ("paidProductPlacementDetails" in calls["body"]) is paid
    if paid:
        assert calls["body"]["paidProductPlacementDetails"] == {"hasPaidProductPlacement": True}
