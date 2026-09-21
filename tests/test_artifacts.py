"""Artifact publication rejects substituted ancestor directories."""

import pytest

from spikenaut_etl.artifacts import write_json


def test_ancestor_symlink_cannot_redirect_publication(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    ancestor = tmp_path / "ancestor"
    ancestor.symlink_to(destination, target_is_directory=True)
    with pytest.raises(OSError):
        write_json(ancestor / "nested" / "manifest.json", {"complete": True})
    assert list(destination.iterdir()) == []


def test_cleanup_keeps_pinned_directory_when_path_is_replaced(tmp_path, monkeypatch):
    from spikenaut_etl import artifacts

    output = tmp_path / "output"
    source = tmp_path / "source"
    output.mkdir()
    source.mkdir()
    (output / "manifest.json").write_text("old output")
    (source / "manifest.json").write_text("source sentinel")
    original = artifacts.os.unlink

    def swap_before_unlink(name, **kwargs):
        output.rename(tmp_path / "detached")
        output.symlink_to(source, target_is_directory=True)
        original(name, **kwargs)

    monkeypatch.setattr(artifacts.os, "unlink", swap_before_unlink)
    with pytest.raises(OSError, match="directory changed"):
        artifacts.clean_artifacts(output, ("manifest.json",))
    assert (source / "manifest.json").read_text() == "source sentinel"
    assert not (tmp_path / "detached/manifest.json").exists()
