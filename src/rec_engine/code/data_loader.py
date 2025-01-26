# DataLoader  module 
# loads the data from desired directory or datastore component in azure data asset

import pandas as pd
from typing import Any, Dict, Union
from pathlib import Path
from src.rec_engine.code.config import get_toml, ParametersConfig, SpecificParametersConfig
from rec_engine.code.config import get_toml


class DataLoader:
    def __init__(
        self, config: Dict[str, Any]
    ) -> None:
        self.config = config
        self.args = ParametersConfig.from_toml(path_or_dict=self.config)
        self.data_loader_args = SpecificParametersConfig.from_toml(path_or_dict=self.config.get('args'), entry="data_loader")
    
    @classmethod
    def from_toml(
        cls,
        path_or_dict: Union[Path, str, Dict[str, Any]],
    ) -> "DataLoader":
        if isinstance(path_or_dict, dict):
            config = path_or_dict
        else:
            config = get_toml()
        return cls(config=config)
    
    def load_data(
        self,
    ) -> pd.DataFrame:
        """
        Load data from data folder
        """
        # check if data folder exists with movie and rating files
        assert Path(self.args.data_folder).exists(), "Data folder does not exist"
        assert Path(f"{self.args.data_folder}/{self.data_loader_args.movie_file}").exists(), "Movie file does not exist"
        assert Path(f"{self.args.data_folder}/{self.data_loader_args.rating_file}").exists(), "Rating file does not exist"
        # read item and rating files
        df_movies = pd.read_csv(f"{self.args.data_folder}/{self.data_loader_args.movie_file}")
        df_ratings = pd.read_csv(f"{self.args.data_folder}/{self.data_loader_args.rating_file}")
        # check if item and rating files are not empty
        assert not df_movies.empty, "Movie file is empty"
        assert not df_ratings.empty, "Rating file is empty"
        # merge item and rating files
        df = pd.merge(df_ratings, df_movies, on=self.args.item_id, how= 'left')
        return df

if __name__ == "__main__":
    config = get_toml()
    dl = DataLoader.from_toml(config)
    df_raw = dl.load_data()