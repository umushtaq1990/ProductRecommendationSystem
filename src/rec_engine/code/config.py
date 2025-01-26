from pathlib import Path
from typing import Any, Dict, Optional, Union, cast
from toml import load
from dataclasses import dataclass, asdict, field
from abc import ABC
from multimethod import multimethod

DictConfig = Dict[str, Any]

__here__ = Path(__file__).resolve()
__root__ = __here__.parents[4]
ConfigPathOrDict = Union[Path, str, Dict[str, Any]]


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

class TOMLConfig(ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    @multimethod
    def from_toml(cls, path_or_dict:ConfigPathOrDict, entry: Optional[str] = None) -> "TOMLConfig": 
        raise NotImplementedError(
            f"No available method for type: {type(path_or_dict)!r}"
        )

@TOMLConfig.from_toml.register
def from_toml_dict(  # type: ignore[no-untyped-def]
    cls,
    path_or_dict: Dict[str, Any],
    entry: Optional[str] = None,
):
    cfg = path_or_dict
    if entry is not None:
        cfg = cfg.get(entry, None)
    return cls(**cfg)

@dataclass(frozen=True)
class BaseFrozenConfig(ABC):
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=False)
class BaseNonFrozenConfig(ABC):
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

@dataclass(init=True, frozen=False)
class ParametersConfig(TOMLConfig, BaseNonFrozenConfig):
    """
    Sets all parameters from args section of config.toml file.

    :param user_id: String containing user_id column name
    :param rating_id: String containing rating_id column name
    :param data_folder: String containing data folder path
    :param data_loader: Dictionary containing data_loader specific parameters
    :param data_processor: Dictionary containing data_processor specific parameters

    """
    user_id: str
    item_id: str
    data_folder: str
    data_loader: Dict[str, Any]
    data_processor: Dict[str, Any]

    def __post_init__(self) -> None:
        self.data_loader = {str(k): str(v) for k, v in self.data_loader.items()}
        self.data_processor = {str(k): str(v) for k, v in self.data_processor.items()}

    @classmethod
    def from_toml(  # type: ignore[no-untyped-def]
        cls, path_or_dict, entry: Optional[str] = "args"
    ) -> "ParametersConfig":
        r = cast(ParametersConfig, super().from_toml(path_or_dict, entry))
        return r
    

@dataclass
class SpecificParametersConfig(ABC):
    param_section: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.param_section.items():
            setattr(self, key, value)

    @classmethod
    def from_toml(cls, path_or_dict: Union[Path, str, Dict[str, Any]], entry: Optional[str] = None) -> "SpecificParametersConfig":
        if isinstance(path_or_dict, dict):
            config = path_or_dict
        else:
            config = get_toml(path_or_dict)
        
        if entry:
            sub_dict = config.get(entry, {})
        else:
            sub_dict = config
        
        return cls(param_section=sub_dict)