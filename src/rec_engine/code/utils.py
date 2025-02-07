import logging
from typing import Any, Dict

import pandas as pd

# Configure logging
from rec_engine.code.logger import LoggerConfig

logger = LoggerConfig.configure_logger("Utils")


def remove_outliers(
    df: pd.DataFrame, col: str, threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers where col is greater than the specified number of standard deviations on the positive side.
    """
    logger.info(f"Removing outliers from {col}")
    mean = df[col].mean()
    std_dev = df[col].std()
    outlier_threshold = mean + threshold * std_dev
    df_filtered = df[df[col] <= outlier_threshold]
    logger.info(
        f"Number of records where {col} is greater than {threshold} standard deviations: "
        f"{df[col].gt(outlier_threshold).sum()}"
    )
    return df_filtered


def scale_min_max(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Scale the col column using min-max scaling.
    """
    logger.info(f"Scaling {col} using min-max scaling")
    min_val = df[col].min()
    max_val = df[col].max()
    df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


def remove_outliers_and_scale(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Remove outliers where col is greater than 3 standard deviations on the positive side
    and scale the col column using min-max scaling.
    """
    try:
        df = remove_outliers(df, col)
        df = scale_min_max(df, col)
        logger.info(f"Outliers removed and {col} scaled successfully")
    except Exception as e:
        logger.error(f"Error in removing outliers and scaling {col}: {e}")
        raise
    return df
