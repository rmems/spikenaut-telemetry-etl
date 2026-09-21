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


@pytest.mark.parametrize("name", ["../outside", "absolute", ".", "..", ""])
def test_cleanup_names_must_be_single_components(tmp_path, name):
    from spikenaut_etl.artifacts import clean_artifacts

    outside = tmp_path / "outside"
    outside.write_text("sentinel")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(OSError, match="single path component"):
        clean_artifacts(output, (str(outside) if name == "absolute" else name,))
    assert outside.read_text() == "sentinel"


def test_overlapping_publishers_cannot_mix_generations(tmp_path):
    from spikenaut_etl.artifacts import pinned_publication

    output = tmp_path / "output"
    artifact = output / "manifest.json"
    with pinned_publication(output):
        write_json(artifact, {"generation": 1})
        original = artifact.read_bytes()
        with pytest.raises(OSError, match="publisher"):
            with pinned_publication(output):
                write_json(artifact, {"generation": 2})
        assert artifact.read_bytes() == original
    with pinned_publication(output):
        write_json(artifact, {"generation": 3})
    assert '"generation": 3' in artifact.read_text()


def test_publisher_lock_released_after_failure(tmp_path):
    from spikenaut_etl.artifacts import pinned_publication

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="fixture failure"):
        with pinned_publication(output):
            raise ValueError("fixture failure")
    with pinned_publication(output):
        write_json(output / "manifest.json", {"complete": True})
