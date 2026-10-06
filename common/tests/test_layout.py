"""Keep the three-folder layout usable without changing historical artifacts."""
from pathlib import Path

from common.src.utils import ROOT, load_config, project_path


def test_repository_root_and_shared_files_are_resolved():
    assert ROOT == Path(__file__).resolve().parents[2]
    assert (ROOT / "common/data/source_manifest.json").is_file()
    assert (ROOT / "common/LICENSE").is_file()
    for model in ("lstm", "gru"):
        config = load_config(f"common/configs/lstm_gru/{model}.yaml")
        assert config["model"] == model
        assert config["processed_dir"] == "lstm_gru/data/processed"
        assert config["output_dir"] == "lstm_gru/runs"


def test_historical_checkpoint_path_is_supported_without_rewriting_weights(tmp_path):
    assert project_path("data/processed") == ROOT / "lstm_gru/data/processed"
    assert project_path("lstm_gru/data/processed") == ROOT / "lstm_gru/data/processed"
    assert project_path(tmp_path) == tmp_path


def test_both_experiments_have_independent_entry_points():
    from lstm_gru.__main__ import main as baseline_main
    from pollen_reproduction.__main__ import main as pollen_main
    assert callable(baseline_main)
    assert callable(pollen_main)
    assert baseline_main is not pollen_main
