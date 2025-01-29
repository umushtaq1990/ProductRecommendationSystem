# DataLoader  module
# loads the data from desired directory or datastore component in azure data asset
import logging
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azureml.core import Dataset, Workspace

from rec_engine.code.config import ParametersConfig, get_toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: Dict[str, Any]) -> None:
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
            config = get_toml()
        return cls(config=config)

    @staticmethod
    def load_data_from_blob(
        account_url: str, container_name: str, item_file: str, rating_file: str
    ) -> pd.DataFrame:
        """
        Load data from Azure Blob Storage
        """
        logger.info("Trying to load data from Azure Blob Storage")
        # Initialize Azure Blob Service Client
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=account_url, credential=credential
        )
        container_client = blob_service_client.get_container_client(
            container_name
        )

        # Download data files
        movie_blob_client = container_client.get_blob_client(item_file)
        movie_data = movie_blob_client.download_blob().readall()
        df_items = pd.read_csv(pd.compat.StringIO(movie_data.decode("utf-8")))

        rating_blob_client = container_client.get_blob_client(rating_file)
        rating_data = rating_blob_client.download_blob().readall()
        df_ratings = pd.read_csv(
            pd.compat.StringIO(rating_data.decode("utf-8"))
        )
        logger.info("Loaded data from Azure Blob Storage")

        # Check if item and rating files are not empty
        assert not df_items.empty, "Movie file is empty"
        assert not df_ratings.empty, "Rating file is empty"

        return df_items, df_ratings

    @staticmethod
    def load_data_from_csv(file_path: str) -> pd.DataFrame:
        """
        Load data from CSV files
        """
        logger.info(f"trying to load data from local file {file_path}")
        assert Path(file_path).exists(), "file does not exist"
        df = pd.read_csv(file_path)
        logger.info("data loaded successfully")
        return df

    def load_data_from_local(
        self, dir_path: str, rating_file: str, item_file: str
    ) -> pd.DataFrame:
        """
        Load data from local files
        """
        logger.info(f"Trying to load data from local directory {dir_path}")
        assert Path(dir_path).exists(), "directory does not exist"
        df_items = self.load_data_from_csv(f"{dir_path}/{item_file}")
        df_ratings = pd.read_csv(f"{dir_path}/{rating_file}")
        return df_items, df_ratings

    def load_data(
        self,
    ) -> pd.DataFrame:
        """
        Load data from Azure ML registered data component, Azure Blob Storage, or data folder
        """
        try:
            # Try to load data from Azure ML registered data component
            logger.info(
                "Trying to load data from Azure ML registered data component"
            )
            ws = Workspace.from_config()
            dataset = Dataset.get_by_name(ws, name="raw_movie_rating_data")
            df = dataset.to_pandas_dataframe()
            logger.info("Loaded data from Azure ML registered data component")
        except Exception as e:
            logger.error(f"Failed to load data from Azure ML: {e}")
            item_file = getattr(self.args.data_loader, "item_file", "NA")
            rating_file = getattr(self.args.data_loader, "rating_file", "NA")
            assert all(
                isinstance(param, str) for param in [item_file, rating_file]
            ), "item and rating file parameters are not provided"
            try:
                # Try to load data from Azure Blob Storage
                account_url = getattr(
                    self.args.data_loader, "azure_account_url", "NA"
                )
                container_name = getattr(
                    self.args.data_loader, "azure_container_name", "NA"
                )
                # Check if all required parameters are provided in string format
                assert all(
                    isinstance(param, str)
                    for param in [account_url, container_name]
                ), "Azure Blob Storage parameters are not provided"

                df_items, df_ratings = self.load_data_from_blob(
                    account_url=account_url,
                    container_name=container_name,
                    item_file=item_file,
                    rating_file=rating_file,
                )
            except Exception as e:
                logger.error(
                    f"Failed to load data from Azure Blob Storage: {e}"
                )
                # Fallback to loading data from data folder
                data_folder = getattr(self.args, "data_folder", "NA")
                assert isinstance(
                    data_folder, str
                ), "data folder path is not provided"
                df_items, df_ratings = self.load_data_from_local(
                    dir_path=data_folder,
                    rating_file=rating_file,
                    item_file=item_file,
                )

            # Check if item and rating files are not empty
            assert not df_items.empty, "Item file is empty"
            assert not df_ratings.empty, "Rating file is empty"
            # Merge item and rating files
            df = pd.merge(
                df_ratings, df_items, on=self.args.item_id, how="left"
            )
            logger.info("Merged movie and rating data")

        return df


if __name__ == "__main__":
    config = get_toml()
    dl = DataLoader.from_toml(config)
    df_raw = dl.load_data()
