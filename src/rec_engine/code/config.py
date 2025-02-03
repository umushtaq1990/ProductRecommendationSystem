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
    az_ws_item_rating_data: str
    item_file: str
    rating_file: str
    output_file: str


class BlobStorageConfig(TypedDict, total=False):
    azure_account_url: str
    azure_container_name: str


class DataProcessorConfig(TypedDict, total=False):
    validation_data_duration: int
    genres_drop_threshold: int
    rating_year_col: str
    duration_release_viewed_col: str
    output_file: str


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
    :param item_id: String containing item_id column name
    :param title_id: String containing title_id column name
    :param date_id: String containing date_id column name
    :param genres_id: String containing genres_id column name
    :param data_folder: String containing data folder path
    :param data_loader: Dictionary containing data_loader specific parameters
    :param data_processor: Dictionary containing data_processor specific parameters

    """

    user_id: str
    item_id: str
    title_id: str
    date_id: str
    genres_id: str
    data_folder: str
    data_loader: DataLoaderConfig
    data_processor: DataProcessorConfig
    blob_params: BlobStorageConfig

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
            title_id=args.get("title_id"),
            date_id=args.get("date_id"),
            genres_id=args.get("genres_id"),
            data_folder=args.get("data_folder"),
            data_loader=args.get("data_loader", {}),
            data_processor=args.get("data_processor", {}),
            blob_params=args.get("blob_params", {}),
        )
