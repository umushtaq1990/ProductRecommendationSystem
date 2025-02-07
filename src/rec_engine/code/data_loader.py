# DataLoader  module
# loads the data from desired directory or datastore component in azure data asset
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from azureml.core import Dataset, Run, Workspace

src_path = Path(__file__).resolve().parents[2]
sys.path.append(str(src_path))

from rec_engine.code.azure_utils import (
    get_blob_container_client,
    get_ws,
    upload_data_frame_to_blob,
)
from rec_engine.code.config import DataLoaderConfig, ParametersConfig, get_toml
from rec_engine.code.logger import LoggerConfig

# Configure logging
logger = LoggerConfig.configure_logger("DataLoader")


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
    def load_data_from_azure_ml(
        ws: Workspace, dataset_name: str
    ) -> pd.DataFrame:
        """
        Load data from Azure ML registered data component.

        :param workspace_config_path: Path to the Azure ML workspace configuration file
        :param dataset_name: Name of the registered dataset
        :return: DataFrame containing the loaded data
        """
        logger.info(
            f"trying to load data from Azure ML registered data component: {dataset_name}"
        )
        dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
        df = dataset.to_pandas_dataframe()
        logger.info(
            f"Loaded data from Azure ML registered data component, shape: {df.shape}"
        )
        return df

    @staticmethod
    def load_data_from_blob(
        account_url: str, container_name: str, item_file: str, rating_file: str
    ) -> pd.DataFrame:
        """
        Load data from Azure Blob Storage
        """
        logger.info("Trying to load data from Azure Blob Storage")
        container_client = get_blob_container_client(
            account_url, container_name
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

    def log_metrics(self, df: pd.DataFrame) -> None:
        """
        Log metrics to Azure ML
        """
        run = Run.get_context()
        run.log("rows", df.shape[0])
        run.log("columns", df.shape[1])
        run.log_list("columns_list", df.columns.tolist())
        # log number of users and items
        run.log("num_users", df[self.args.user_id].nunique())
        run.log("num_items", df[self.args.item_id].nunique())
        logger.info("Logged metrics")

    def load_data(
        self,
    ) -> pd.DataFrame:
        """
        Load data from Azure ML registered data component, Azure Blob Storage, or data folder
        """
        logger.info("Loading data")

        output_dir = self.args.data_folder

        if not self.args.local_run:
            logger.info("Running in Azure ML environment")
            data_available_in_ws = False
            ws, run_context_available = get_ws()
            try:
                # Try to load data from Azure ML registered data component
                df = self.load_data_from_azure_ml(
                    ws=ws,
                    dataset_name=getattr(
                        self.args.data_loader, "az_ws_item_rating_data", "NA"
                    ),
                )
                data_available_in_ws = True
            except Exception as e:
                logger.error(f"Failed to load data from Azure ML: {e}")
                try:
                    # Try to load data from Azure Blob Storage
                    df_items, df_ratings = self.load_data_from_blob(
                        account_url=getattr(
                            self.args.blob_params, "azure_account_url", "NA"
                        ),
                        container_name=getattr(
                            self.args.blob_params, "azure_container_name", "NA"
                        ),
                        item_file=getattr(
                            self.args.data_loader, "item_file", "NA"
                        ),
                        rating_file=getattr(
                            self.args.data_loader, "rating_file", "NA"
                        ),
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load data from Azure Blob Storage: {e}"
                    )
                    # Fallback to loading data from data folder
                    df_items, df_ratings = self.load_data_from_local(
                        dir_path=self.args.data_folder,
                        rating_file=getattr(
                            self.args.data_loader, "rating_file", "NA"
                        ),
                        item_file=getattr(
                            self.args.data_loader, "item_file", "NA"
                        ),
                    )

                    # upload item, rating data to blob storage
                    upload_data_frame_to_blob(
                        account_url=getattr(
                            self.args.blob_params, "azure_account_url", "NA"
                        ),
                        container_name=getattr(
                            self.args.blob_params, "azure_container_name", "NA"
                        ),
                        file_name=getattr(
                            self.args.data_loader, "item_file", "NA"
                        ),
                        df=df_items,
                    )
                    upload_data_frame_to_blob(
                        account_url=getattr(
                            self.args.blob_params, "azure_account_url", "NA"
                        ),
                        container_name=getattr(
                            self.args.blob_params, "azure_container_name", "NA"
                        ),
                        file_name=getattr(
                            self.args.data_loader, "rating_file", "NA"
                        ),
                        df=df_ratings,
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
                    dataset = Dataset.Tabular.register_pandas_dataframe(
                        dataframe=df,
                        target=ws.get_default_datastore(),
                        name=getattr(
                            self.args.blob_params,
                            "az_ws_item_rating_data",
                            "NA",
                        ),
                        description="Item and rating data",
                        tags={"source": "Azure Blob Storage"},
                    )
                    logger.info(
                        "Saved data to Azure ML registered data component"
                    )

            # Log metrics to Azure ML
            if run_context_available:
                self.log_metrics(df)
                output_dir = os.path.join(
                    os.environ["AZUREML_DATAREFERENCE_outputs"]
                )
        else:
            logger.info("Running in local environment")
            # Load data from local directory
            df_items, df_ratings = self.load_data_from_local(
                dir_path=self.args.data_folder,
                rating_file=getattr(self.args.data_loader, "rating_file", "NA"),
                item_file=getattr(self.args.data_loader, "item_file", "NA"),
            )
            # Check if item and rating files are not empty
            assert not df_items.empty, "Item file is empty"
            assert not df_ratings.empty, "Rating file is empty"
            # Merge item and rating files
            df = pd.merge(
                df_ratings, df_items, on=self.args.item_id, how="left"
            )
            logger.info("Merged item and rating data")

        os.makedirs(output_dir, exist_ok=True)
        # create file path
        file_path = Path(
            output_dir, getattr(self.args.data_loader, "output_file", "NA")
        )
        df.to_pickle(file_path)
        logger.info(f"raw data saved to {file_path}")
        return df


if __name__ == "__main__":
    config = get_toml()
    dl = DataLoader.from_toml(config)
    df_raw = dl.load_data()
