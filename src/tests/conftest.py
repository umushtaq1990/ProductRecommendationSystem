from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from rec_engine.code.config import get_toml
from rec_engine.code.data_loader import DataLoader
from rec_engine.code.data_processor import DataProcessor


@pytest.fixture(autouse=True)
def src_path() -> Path:
    """Path to the src folder"""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def data_path() -> Path:
    """Path to the data folder"""
    return Path(__file__).resolve().parents[3].joinpath("data")


@pytest.fixture(autouse=True)
def test_data_dir_path() -> Path:
    """Path to the test data directory"""
    return Path(__file__).resolve().parent.joinpath(".test_data")


@pytest.fixture(autouse=True)
def config_path(src_path: Path) -> Path:
    """Path to the configuration file"""
    return src_path.joinpath("rec_engine/config.toml")


@pytest.fixture(autouse=True)
def config(config_path: Path, test_data_dir_path: Path) -> Dict[str, Any]:
    """Read config file"""
    conf = get_toml(path=config_path)
    # assign .test_data folder in test directory to data_folder
    conf["args"]["data_folder"] = test_data_dir_path
    return conf


@pytest.fixture
def data_loader_module(config: Dict[str, Any]) -> DataLoader:
    """get data loader class object"""
    return DataLoader.from_toml(path_or_dict=config)


@pytest.fixture
def data_processor_module(config: Dict[str, Any]) -> DataProcessor:
    """get data loader class object"""
    return DataProcessor.from_toml(path_or_dict=config)


@pytest.fixture
def raw_data(
    data_loader_module: DataLoader,
) -> pd.DataFrame:
    """
    test data loading
    """
    raw_data = data_loader_module.load_data()
    return raw_data
