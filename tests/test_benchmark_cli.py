"""Reports must never replace another run, including races after validation."""

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import benchmark_timesfm as cli
from tests.conftest import make_ohlcv


@pytest.mark.parametrize("existing", ["archive", "json", "alias", "dangling_symlink"])
def test_refuses_conflicting_outputs_before_fetch_or_inference(tmp_path, monkeypatch, existing):
    archive, output = tmp_path / "run.zip", tmp_path / "run.json"
    if existing == "archive":
        archive.write_bytes(b"original")
    if existing == "json":
        output.write_bytes(b"original")
    if existing == "alias":
        output = archive
    if existing == "dangling_symlink":
        output.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(sys, "argv", ["benchmark", "TEST", "--test-start", "2025-01-01",
                                     "--archive", str(archive), "--output", str(output)])
    def fail(*args, **kwargs):
        pytest.fail("Invalid output paths must fail before network/model work")
    monkeypatch.setattr(cli, "fetch_ohlcv", fail)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    if existing in {"archive", "json"}:
        assert (archive if existing == "archive" else output).read_bytes() == b"original"


def test_json_created_by_another_process_is_not_overwritten(tmp_path, monkeypatch):
    archive, output = tmp_path / "run.zip", tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "TEST", "--test-start", "2024-04-01",
                                     "--archive", str(archive), "--output", str(output)])
    monkeypatch.setattr(cli, "fetch_ohlcv", lambda *args, **kwargs: make_ohlcv(200))
    def benchmark(*args, **kwargs):
        output.write_text(json.dumps({"original": True}))
        return SimpleNamespace(configuration={}, to_dict=lambda: {"new": True})
    monkeypatch.setattr(cli, "benchmark_close_series", benchmark)
    monkeypatch.setattr(cli, "build_archive", lambda *args, **kwargs: b"valid run archive")
    with pytest.raises(FileExistsError):
        cli.main()
    assert json.loads(output.read_text()) == {"original": True}
    assert archive.read_bytes() == b"valid run archive"
