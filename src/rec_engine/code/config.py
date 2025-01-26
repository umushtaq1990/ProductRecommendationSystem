from pathlib import Path
from typing import Any, Dict, Optional, Union
from toml import load


DictConfig = Dict[str, Any]

__here__ = Path(__file__).resolve()
__root__ = __here__.parents[4]


def get_toml(path: Optional[Union[Path, str]] = None) -> DictConfig:
    """Read TOML configuration"""
    if path is not None:
        path = Path(path).resolve()
    else:
        path = Path(__file__).parent.joinpath(f"../config.toml")
    assert (path.exists() and path.is_file() and path.suffix.endswith(".toml")), f"Invalid config path provided: {str(path)}"

    with open(path, mode="r", encoding="utf-8") as file:
        cfg = load(file)
    return cfg