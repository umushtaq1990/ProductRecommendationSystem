import logging

import pandas as pd
from azureml.core import Dataset, Workspace

from rec_engine.code.logger import LoggerConfig

# Configure logging
logger = LoggerConfig.configure_logger("AzureUtils")


def register_or_update_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Register or update the dataset in Azure ML
    """
    try:
        ws = Workspace.from_config()
        dataset = Dataset.get_by_name(ws, dataset_name)
        logger.info("Data already registered as a dataset in Azure ML")
        # if dataset exists, update the dataset if the data has changed
        if (
            dataset.get_profile().compute_digest(df)
            != dataset.get_profile().compute_digest()
        ):
            dataset.update_from_dataframe(df)
            logger.info("Data updated in Azure ML")
    except Exception as e:
        try:
            ws = Workspace.from_config()
            dataset = Dataset.Tabular.register_pandas_dataframe(
                df, ws.get_default_datastore(), dataset_name
            )
            logger.info("Data registered as a dataset in Azure ML")
        except Exception as e:
            logger.error(
                f"Error registering data as a dataset in Azure ML: {e}"
            )
