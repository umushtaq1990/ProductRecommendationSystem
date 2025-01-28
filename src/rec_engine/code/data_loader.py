# DataLoader  module 
# loads the data from desired directory or datastore component in azure data asset
import logging
from azureml.core import Workspace, Dataset
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import pandas as pd
from typing import Any, Dict, Union
from pathlib import Path
from src.rec_engine.code.config import get_toml, ParametersConfig, SpecificParametersConfig
from rec_engine.code.config import get_toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def load_data_from_blob(self) -> pd.DataFrame:
        """
        Load data from Azure Blob Storage
        """
        logger.info("Trying to load data from Azure Blob Storage")
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=self.data_loader_args.azure_account_url, credential=credential)
        container_client = blob_service_client.get_container_client(self.data_loader_args.azure_container_name)
        
        blob_client_movies = container_client.get_blob_client("raw_movie_rating_data/movies.csv")
        blob_client_ratings = container_client.get_blob_client("raw_movie_rating_data/ratings.csv")
        
        df_movies = pd.read_csv(blob_client_movies.download_blob().content_as_text())
        df_ratings = pd.read_csv(blob_client_ratings.download_blob().content_as_text())
        logger.info("Loaded data from Azure Blob Storage")

        # Check if item and rating files are not empty
        assert not df_movies.empty, "Movie file is empty"
        assert not df_ratings.empty, "Rating file is empty"

        return df_movies, df_ratings

    def load_data_from_local(self) -> pd.DataFrame:
        """
        Load data from local CSV files
        """
        logger.info("Trying to load data from local CSV files")
        assert Path(self.args.data_folder).exists(), "Data folder does not exist"
        assert Path(f"{self.args.data_folder}/{self.data_loader_args.movie_file}").exists(), "Movie file does not exist"
        assert Path(f"{self.args.data_folder}/{self.data_loader_args.rating_file}").exists(), "Rating file does not exist"
        df_movies = pd.read_csv(f"{self.args.data_folder}/{self.data_loader_args.movie_file}")
        df_ratings = pd.read_csv(f"{self.args.data_folder}/{self.data_loader_args.rating_file}")
        logger.info("Loaded data from local CSV files")
        
        return df_movies, df_ratings
    
    def load_data(
        self,
    ) -> pd.DataFrame:
        """
        Load data from Azure ML registered data component, Azure Blob Storage, or data folder
        """
        try:
            # Try to load data from Azure ML registered data component
            logger.info("Trying to load data from Azure ML registered data component")
            ws = Workspace.from_config()
            dataset = Dataset.get_by_name(ws, name="raw_movie_rating_data")
            df = dataset.to_pandas_dataframe()
            logger.info("Loaded data from Azure ML registered data component")
        except Exception as e:
            logger.error(f"Failed to load data from Azure ML: {e}")
            try:
                # Try to load data from Azure Blob Storage
                df_movies, df_ratings = self.load_data_from_blob()
            except Exception as e:
                logger.error(f"Failed to load data from Azure Blob Storage: {e}")
                # Fallback to loading data from data folder
                df_movies, df_ratings = self.load_data_from_local()

            # Check if item and rating files are not empty
            assert not df_movies.empty, "Movie file is empty"
            assert not df_ratings.empty, "Rating file is empty"
            # Merge item and rating files
            df = pd.merge(df_ratings, df_movies, on=self.args.item_id, how='left')
            logger.info("Merged movie and rating data")

        return df


if __name__ == "__main__":
    config = get_toml()
    dl = DataLoader.from_toml(config)
    df_raw = dl.load_data()