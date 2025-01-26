# DataLoader  module 
# loads the data from desired directory or datastore component in azure data asset

import pandas as pd
from azureml.core import Dataset, Datastore, Workspace
from .config import get_toml


class DataLoader:
    def __init__(
        self, args: ParametersConfig,
    ) -> None:
        self.args = args
    
    @classmethod
    def from_toml(cls) -> "DataLoader":
        return cls(config=get_toml())
    
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
