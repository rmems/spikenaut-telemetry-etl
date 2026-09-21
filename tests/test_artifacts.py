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
