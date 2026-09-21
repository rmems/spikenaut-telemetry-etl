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


@pytest.mark.parametrize("create_only", [False, True])
def test_publication_syncs_file_before_install_and_directory_after(
    tmp_path, monkeypatch, create_only
):
    from spikenaut_etl import artifacts

    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "manifest.json"
    events = []
    original_fsync = artifacts.os.fsync
    original_link = artifacts.os.link
    original_replace = artifacts.os.replace

    def record_fsync(descriptor):
        events.append("fsync")
        original_fsync(descriptor)

    def record_link(*args, **kwargs):
        events.append("publish")
        original_link(*args, **kwargs)

    def record_replace(*args, **kwargs):
        events.append("publish")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(artifacts.os, "fsync", record_fsync)
    monkeypatch.setattr(artifacts.os, "link", record_link)
    monkeypatch.setattr(artifacts.os, "replace", record_replace)

    if create_only:
        with artifacts.no_replace_publication():
            write_json(artifact, {"complete": True})
    else:
        write_json(artifact, {"complete": True})

    assert events == ["fsync", "publish", "fsync"]


def test_directory_creation_syncs_each_parent(tmp_path, monkeypatch):
    from spikenaut_etl import artifacts

    events = []
    original_fsync = artifacts.os.fsync
    original_mkdir = artifacts.os.mkdir

    def record_fsync(descriptor):
        events.append("fsync")
        original_fsync(descriptor)

    def record_mkdir(name, *args, **kwargs):
        original_mkdir(name, *args, **kwargs)
        events.append(f"mkdir:{name}")

    monkeypatch.setattr(artifacts.os, "fsync", record_fsync)
    monkeypatch.setattr(artifacts.os, "mkdir", record_mkdir)

    write_json(tmp_path / "output" / "nested" / "manifest.json", {"complete": True})

    created = [index for index, event in enumerate(events) if event.startswith("mkdir:")]
    assert [events[index] for index in created] == ["mkdir:output", "mkdir:nested"]
    assert all(events[index + 1] == "fsync" for index in created)
