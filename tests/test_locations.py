"""msfm.utils.locations: the two dataset files of the repository, the validation and the CLI."""

import os

import pytest

from msfm.utils import locations

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DESY6 = os.path.join(REPO_DIR, "configs/desy6/v1/v1.yaml")
DESY6_V2 = os.path.join(REPO_DIR, "configs/desy6/v2/v2.yaml")
DESY3 = os.path.join(REPO_DIR, "configs/desy3/v18/v18.yaml")


def _dataset(tmp_path, text):
    """A config next to a locations file with the given text. Returns the config path."""
    (tmp_path / "locations.yaml").write_text(text)
    config = tmp_path / "v1.yaml"
    config.write_text("survey: {name: TEST}\n")
    return str(config)


def test_desy6_resolves_the_production_paths():
    resolved = locations.load(DESY6)
    assert resolved["data_dir"] == "/ritom/scratch/cscs/athomsen/desy6/v1/default"
    assert resolved["obs_dir"] == "/ritom/scratch/cscs/athomsen/desy6/v1/default/obs"
    assert resolved["archive_dir"] == "/capstor/store/cscs/userlab/lp184/athomsen/deep_lss/data/desy6/v1/default"
    assert resolved["runs_dir"] == "/ritom/scratch/cscs/athomsen/deep_lss/runs/desy6/v1/default"
    assert resolved["projection"].endswith("/processed/desy6/CosmoGrid/bary")


def test_desy3_resolves_the_v18_paths():
    resolved = locations.load(DESY3)
    assert resolved["data_dir"] == "/ritom/scratch/cscs/athomsen/v18/default"
    assert resolved["obs_dir"] == "/capstor/store/cscs/swissai/a0158/athomsen/deep_lss/data/v18/default/obs"
    assert resolved["runs_dir"] == "/capstor/store/cscs/swissai/a0158/athomsen/deep_lss/runs/v18/default"
    assert "projection" not in resolved


def test_desy6_v2_resolves_its_roots_directly():
    resolved = locations.load(DESY6_V2)
    assert resolved["data_dir"] == "/ritom/scratch/cscs/athomsen/desy6/v2"
    assert resolved["obs_dir"] == "/ritom/scratch/cscs/athomsen/desy6/v2/obs"
    assert resolved["runs_dir"] == "/ritom/scratch/cscs/athomsen/deep_lss/runs/desy6/v2"


def test_dataset_config_is_named_after_its_version():
    assert locations.dataset_config("desy6/v1") == DESY6
    assert locations.dataset_config("desy3/v18/") == DESY3
    assert os.path.isfile(locations.dataset_config("desy6/v2"))


def test_obs_defaults_to_data(tmp_path):
    assert locations.load(_dataset(tmp_path, "data: /d\n")) == {"data_dir": "/d", "obs_dir": "/d/obs"}


def test_empty_file_resolves_nothing(tmp_path):
    assert locations.load(_dataset(tmp_path, "")) == {}


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(KeyError, match="unknown keys"):
        locations.load(_dataset(tmp_path, "data: /d\ndatta: /e\n"))


@pytest.mark.parametrize("value", ["relative/path", "~/home", "3", "[/a, /b]"])
def test_value_must_be_an_absolute_path(tmp_path, value):
    with pytest.raises(ValueError, match="absolute path"):
        locations.load(_dataset(tmp_path, f"data: {value}\n"))


def test_missing_file_is_an_error(tmp_path):
    config = tmp_path / "v1.yaml"
    config.write_text("")
    with pytest.raises(FileNotFoundError, match="locations.yaml does not exist"):
        locations.load(str(config))


def test_get_names_the_file_when_a_key_is_missing():
    with pytest.raises(KeyError, match="desy3/v18/locations.yaml gives no projection"):
        locations.get(DESY3, "projection")


def test_cli_prints_every_path_or_one(capsys):
    locations.main([DESY6])
    lines = capsys.readouterr().out.splitlines()
    assert "data_dir=/ritom/scratch/cscs/athomsen/desy6/v1/default" in lines
    assert all("=" in line for line in lines)

    locations.main([DESY6, "--key", "obs_dir"])
    assert capsys.readouterr().out == "/ritom/scratch/cscs/athomsen/desy6/v1/default/obs\n"


def test_cli_exits_1_on_a_missing_key(capsys):
    with pytest.raises(SystemExit) as exit_info:
        locations.main([DESY3, "--key", "projection"])
    assert exit_info.value.code == 1
    assert "gives no projection" in capsys.readouterr().err
