# DataLoader  module
# loads the data from desired directory or datastore component in azure data asset
import io
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
    def load_data_from_azure_ml(dataset_name: str) -> pd.DataFrame:
        """
        Load data from Azure ML registered data component.

        :param workspace_config_path: Path to the Azure ML workspace configuration file
        :param dataset_name: Name of the registered dataset
        :return: DataFrame containing the loaded data
        """
        logger.info(
            "Trying to load data from Azure ML registered data component"
        )
        ws = Workspace.from_config()
        dataset = Dataset.get_by_name(ws, name=dataset_name)
        df = dataset.to_pandas_dataframe()
        logger.info("Loaded data from Azure ML registered data component")
        return df

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
        item_data = movie_blob_client.download_blob().readall()
        df_items = pd.read_csv(io.StringIO(item_data.decode("utf-8")))

        rating_blob_client = container_client.get_blob_client(rating_file)
        rating_data = rating_blob_client.download_blob().readall()
        df_ratings = pd.read_csv(io.StringIO(rating_data.decode("utf-8")))
        logger.info("Loaded data from Azure Blob Storage")

        # Check if item and rating files are not empty
        assert not df_items.empty, "Movie file is empty"
        assert not df_ratings.empty, "Rating file is empty"

        return df_items, df_ratings

    @staticmethod
    def upload_data_to_blob(
        account_url: str,
        container_name: str,
        item_file: str,
        rating_file: str,
        df_items: pd.DataFrame,
        df_ratings: pd.DataFrame,
    ) -> None:
        """
        Upload data to Azure Blob Storage.

        :param account_url: Azure Blob Storage account URL
        :param container_name: Azure Blob Storage container name
        :param item_file: Name of the item file to upload
        :param rating_file: Name of the rating file to upload
        :param df_items: DataFrame containing item data
        :param df_ratings: DataFrame containing rating data
        """
        logger.info("Uploading data to Azure Blob Storage")

        # Initialize Azure Blob Service Client
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=account_url, credential=credential
        )
        container_client = blob_service_client.get_container_client(
            container_name
        )

        # Upload data files
        movie_blob_client = container_client.get_blob_client(item_file)
        movie_blob_client.upload_blob(
            df_items.to_csv(index=False), overwrite=True
        )
        rating_blob_client = container_client.get_blob_client(rating_file)
        rating_blob_client.upload_blob(
            df_ratings.to_csv(index=False), overwrite=True
        )

        logger.info("Uploaded data to Azure Blob Storage")

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
        data_available_in_ws = False
        try:
            # Try to load data from Azure ML registered data component
            logger.info(
                "Trying to load data from Azure ML registered data component"
            )
            item_rating_data = getattr(
                self.args.data_loader, "az_ws_item_rating_data", "NA"
            )
            df = self.load_data_from_azure_ml(dataset_name=item_rating_data)
            data_available_in_ws = True
        except Exception as e:
            logger.error(f"Failed to load data from Azure ML: {e}")
            item_file = getattr(self.args.data_loader, "item_file", "NA")
            rating_file = getattr(self.args.data_loader, "rating_file", "NA")
            account_url = getattr(
                self.args.data_loader, "azure_account_url", "NA"
            )
            container_name = getattr(
                self.args.data_loader, "azure_container_name", "NA"
            )
            try:
                # Try to load data from Azure Blob Storage
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
                df_items, df_ratings = self.load_data_from_local(
                    dir_path=data_folder,
                    rating_file=rating_file,
                    item_file=item_file,
                )

                # upload item, rating data to blob storage
                self.upload_data_to_blob(
                    account_url=account_url,
                    container_name=container_name,
                    item_file=item_file,
                    rating_file=rating_file,
                    df_items=df_items,
                    df_ratings=df_ratings,
                )

            # Check if item and rating files are not empty
            assert not df_items.empty, "Item file is empty"
            assert not df_ratings.empty, "Rating file is empty"
            # Merge item and rating files
            df = pd.merge(
                df_ratings, df_items, on=self.args.item_id, how="left"
            )
            logger.info("Merged item and rating data")

            # Save data to Azure ML registered data component, if it is not already available
            if not data_available_in_ws:
                logger.info(
                    "Trying to save data to Azure ML registered data component"
                )
                ws = Workspace.from_config()
                dataset = Dataset.Tabular.register_pandas_dataframe(
                    dataframe=df,
                    target=ws.get_default_datastore(),
                    name=item_rating_data,
                    description="Item and rating data",
                    tags={"source": "Azure Blob Storage"},
                )
                logger.info("Saved data to Azure ML registered data component")
        return df


if __name__ == "__main__":
    config = get_toml()
    dl = DataLoader.from_toml(config)
    df_raw = dl.load_data()
