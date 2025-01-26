# DataLoader  module 
# loads the data from desired directory or datastore component in azure data asset

import pandas as pd
from typing import Any, Dict, Union
from pathlib import Path
from config import get_toml, ParametersConfig


class DataLoader:
    def __init__(
        self, config: Dict[str, Any]
    ) -> None:
        self.config = config
        self.args = ParametersConfig.from_toml(path_or_dict=self.config)
    
    @classmethod
    def from_toml(
        cls,
        path_or_dict: Union[Path, str, Dict[str, Any]],
    ) -> "DataLoader":
        if isinstance(path_or_dict, dict):
            config = path_or_dict
        else:
            config = get_toml(config_suffix=str(path_or_dict))
        return cls(config=config)
    
    def load_data(
        self,
    ) -> str:
        """Get the latest folder from a list of folders.

        Args:
            folder_list (List[str]): List of folders and files
            folder_type (Literal["internal", "other"]): Type of folder to get. Use 'other' if structure in a format data_fn/yyyy-mm-dd/files

        Returns:
            str: Latest folder name
        """
        df = pd.read_csv(self.args.data_loader["data_path"])
        return df


if __name__ == "__main__":
    config = get_toml()
    dl = DataLoader.from_toml(config)
    df_raw = dl.load_data()