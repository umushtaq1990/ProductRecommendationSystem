from abc import ABC
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict, Union

from multimethod import multimethod
from toml import load

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
    assert (
        path.exists() and path.is_file() and path.suffix.endswith(".toml")
    ), f"Invalid config path provided: {str(path)}"

    with open(path, mode="r", encoding="utf-8") as file:
        cfg = load(file)
    return cfg


class TOMLConfig(ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    @multimethod
    def from_toml(
        cls, path_or_dict: ConfigPathOrDict, entry: Optional[str] = None
    ) -> "TOMLConfig":
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


class DataLoaderConfig(TypedDict, total=False):
    azure_account_url: str
    azure_container_name: str
    item_file: str
    rating_file: str


class DataProcessorConfig(TypedDict, total=False):
    validation_data_duration: int


@dataclass
class GenericConfig(ABC):
    param_section: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.param_section.items():
            setattr(self, key, value)

    @classmethod
    def from_toml(
        cls, path_or_dict: Dict[str, Any], entry: Optional[str] = None
    ) -> "GenericConfig":
        if entry:
            sub_dict = path_or_dict.get(entry, {})
        else:
            sub_dict = path_or_dict
        return cls(param_section=sub_dict)


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
    data_loader: GenericConfig
    data_processor: GenericConfig

    @classmethod
    def from_toml(
        cls, path_or_dict: ConfigPathOrDict, entry: Optional[str] = "args"
    ) -> "ParametersConfig":
        if isinstance(path_or_dict, (Path, str)):
            config = get_toml(path_or_dict)
        else:
            config = path_or_dict
        entry = entry or "args"  # Ensure entry is a string
        args = config.get(entry, {})
        return cls(
            user_id=args.get("user_id"),
            item_id=args.get("item_id"),
            data_folder=args.get("data_folder"),
            data_loader=GenericConfig.from_toml(args, entry="data_loader"),
            data_processor=GenericConfig.from_toml(
                args, entry="data_processor"
            ),
        )
