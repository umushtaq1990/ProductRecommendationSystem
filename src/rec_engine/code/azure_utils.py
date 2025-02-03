from typing import Tuple

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from azureml.core import Dataset, Run, Workspace

from rec_engine.code.logger import LoggerConfig

# Configure logging
logger = LoggerConfig.configure_logger("AzureUtils")


def get_ws() -> Tuple[Workspace, bool]:
    run_context_available = False
    try:
        run = Run.get_context()
        ws = run.experiment.workspace
        logger.info(f"ws from run {ws}")
        run_context_available = True
    except Exception as e:
        logger.error(f"Failed to get workspace from run: {e}")
        ws = Workspace.from_config()
        logger.info(f"ws from config {ws}")
    return ws, run_context_available


def get_blob_container_client(
    account_url: str, container_name: str
) -> ContainerClient:

    # Initialize Azure Blob Service Client
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=credential
    )
    container_client = blob_service_client.get_container_client(container_name)
    return container_client


def upload_data_frame_to_blob(
    account_url: str,
    container_name: str,
    file_name: str,
    df: pd.DataFrame,
) -> None:
    """
    Upload data to Azure Blob Storage.

    :param account_url: Azure Blob Storage account URL
    :param container_name: Azure Blob Storage container name
    :param file_name: Name of the file where the data will be uploaded
    :param df: DataFrame containing the data to be uploaded
    """
    logger.info("Uploading data to Azure Blob Storage")

    # Initialize Azure Blob Service Client
    container_client = get_blob_container_client(account_url, container_name)
    item_blob_client = container_client.get_blob_client(file_name)
    # if parquet file, use to_parquet method
    if file_name.endswith(".parquet"):
        item_blob_client.upload_blob(df.to_parquet(), overwrite=True)
    elif file_name.endswith(".csv"):
        item_blob_client.upload_blob(df.to_csv(index=False), overwrite=True)
    elif file_name.endswith(".pkl"):
        item_blob_client.upload_blob(df.to_pickle(), overwrite=True)
    else:
        raise ValueError(
            f"Unsupported file format: {file_name}. Supported formats are .parquet and .csv"
        )
    logger.info(
        f"{container_name}/{file_name} Uploaded data to Azure Blob Storage"
    )


def register_or_update_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Register or update the dataset in Azure ML
    """
    try:
        ws, _ = get_ws()
        dataset = Dataset.get_by_name(ws, dataset_name)
        logger.info("Data already registered as a dataset in Azure ML")
        # check if datset profile is available, if not create one
        # TODO check if profile available of registered dataset
        # if not, create one
        try:
            profile = dataset.get_profile()
            logger.info("Dataset profile already exists")
        except Exception as e:
            # Generate dataset profile if it doesn't exist
            dataset_profile = dataset.profile()
            logger.info("Dataset profile created")
        # if dataset exists, update the dataset if the data has changed
        if (
            dataset.get_profile().compute_digest(df)
            != dataset.get_profile().compute_digest()
        ):
            dataset.update_from_dataframe(df)
            logger.info("Data updated in Azure ML")
    except Exception as e:
        try:
            dataset = Dataset.Tabular.register_pandas_dataframe(
                df, ws.get_default_datastore(), dataset_name
            )
            logger.info("Data registered as a dataset in Azure ML")
        except Exception as e:
            logger.error(
                f"Error registering data as a dataset in Azure ML: {e}"
            )
