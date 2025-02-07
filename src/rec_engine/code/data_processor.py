import os
import sys
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from azureml.core import Run

src_path = Path(__file__).resolve().parents[2]
sys.path.append(str(src_path))

from rec_engine.code.azure_utils import (
    get_ws,
    register_or_update_dataset,
    upload_data_frame_to_blob,
)
from rec_engine.code.config import ParametersConfig, get_toml
from rec_engine.code.logger import LoggerConfig

# Configure logging
logger = LoggerConfig.configure_logger("DataProcessor")


class DataProcessor:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.args = ParametersConfig.from_toml(path_or_dict=self.config)

    @classmethod
    def from_toml(
        cls,
        path_or_dict: Union[Path, str, Dict[str, Any]],
    ) -> "DataProcessor":
        if isinstance(path_or_dict, dict):
            config = path_or_dict
        else:
            config = get_toml(path_or_dict)
        return cls(config=config)

    @staticmethod
    def process_genres(
        df: pd.DataFrame, item_id: str, genres_id: str, threshold: int = 5
    ) -> pd.DataFrame:
        """
        Process the genres column in the DataFrame
        """
        logger.info("Processing genres")
        # Get unique items and genres
        df_items = df[[item_id, genres_id]].drop_duplicates()
        # Split genres into multiple columns, so that each genre has its own column
        df_genres = df_items[genres_id].str.get_dummies()
        # Rename (no genres listed) to "no_genres"
        df_genres = df_genres.rename(
            columns={"(no genres listed)": f"no_{genres_id}"}
        )
        # if genres columns contain less than 20 percent of records then drop the columns
        cols_to_drop = []
        for col in df_genres.columns:
            if (df_genres[col].sum() * 100 / df_genres.shape[0]) < threshold:
                cols_to_drop.append(col)
        df_genres = df_genres.drop(columns=cols_to_drop)
        logger.info(
            f"genres dropped as they appeared in few items: {cols_to_drop}"
        )
        # log the columns dropped
        df_items = df_items.join(df_genres)
        # Drop the original genres column
        df_items = df_items.drop(columns=[genres_id])
        # merge the processed genres back to the original dataframe
        df = df.merge(df_items, on=item_id)
        # Drop the original genres column
        df = df.drop(columns=[genres_id])
        logger.info("Genres processed successfully")
        return df

    @staticmethod
    def remove_outliers_and_scale(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Remove outliers where col is greater than 3 standard deviations on the positive side
        and scale the col column using min-max scaling
        """
        logger.info(f"Removing outliers and scaling {col}")
        # Remove outliers where col is greater than 3 standard deviations on the positive side
        # TODO adjust outlier removal logic and move to utils
        # TODO adjust scaling logic and move to utis
        df = df[df[col] <= df[col].mean() + 3 * df[col].std()]
        # Log number of records where col is greater than 3 standard deviations on the positive side
        logger.info(
            f"Number of records where {col} is greater than 3 standard deviations on the positive side: {df[col].gt(df[col].mean() + 3 * df[col].std()).sum()}"
        )
        # Scale the col column using min-max scaling
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        logger.info(f"Outliers removed and {col} scaled successfully")
        return df

    @staticmethod
    def get_years_since_release_viewed_feat(
        df: pd.DataFrame, title_col: str, timestamp_col: str, col_name: str
    ) -> pd.DataFrame:
        """
        Process the title and timestamp columns to generate new columns related to the release year and the year watched
        """
        logger.info("Processing years")
        # Generate a new column "year_released" from the title column
        df["year_released"] = df[title_col].str.extract(r"\((\d{4})\)$")
        df["year_released"] = pd.to_numeric(
            df["year_released"], errors="coerce"
        )
        # Generate year column from timestamp
        df["date_time"] = pd.to_datetime(df[timestamp_col], unit="s")
        df["year_watched"] = df["date_time"].dt.year
        # Get difference between year_released and year_watched
        df[col_name] = df["year_watched"] - df["year_released"]
        # log number of records where col is missing
        logger.info(
            f"Number of records where {col_name} is missing: {df[col_name].isnull().sum()}"
        )
        # Drop records where col is missing
        df = df.dropna(subset=[col_name])
        # remove records where col is negative
        df = df[df[col_name] >= 0]
        # log number of records where col is negative
        logger.info(
            f"Number of records where {col_name} is negative: {df[col_name].lt(0).sum()}"
        )
        # drop date_time column
        df = df.drop(columns=["date_time"])
        logger.info("Years processed successfully")
        return df

    @staticmethod
    def split_train_test(
        df: pd.DataFrame, last_n_years: int, year_col: str
    ) -> pd.DataFrame:
        """
        Split the DataFrame into training and testing datasets based on the year_watched column
        """
        logger.info("Splitting data into train and test sets")
        # Keep last n years data as test data
        test_data = df[df[year_col] >= df[year_col].max() - last_n_years]
        test_data["train"] = False
        # Keep all data except last n years data as train data
        train_data = df[df[year_col] < df[year_col].max() - last_n_years]
        train_data["train"] = True
        # Join train and test data
        df = pd.concat([train_data, test_data])
        logger.info("Data split into train and test sets successfully")
        return df

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

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data
        """
        logger.info("Processing data")
        _, run_context_available = get_ws()
        if run_context_available:
            input_dir = os.environ["AZUREML_DATAREFERENCE_outputs"]
            file_path = os.path.join(
                input_dir, getattr(self.args.data_loader, "output_file", "NA")
            )
        else:
            input_dir = self.args.data_folder
            file_path = str(
                Path(
                    input_dir,
                    getattr(self.args.data_loader, "output_file", "NA"),
                )
            )
        logger.info(f"Input directory: {input_dir}")
        # check if the dataframe is empty
        if df.empty:
            logger.error("Dataframe is empty")
            # try to load the data from the data folder
            try:
                logger.info(f"trying to load data from local file {file_path}")
                assert Path(file_path).exists(), "file does not exist"
                df = pd.read_pickle(file_path)
            except Exception as e:
                logger.error(
                    f"Failed to load the data from the data folder: {e}"
                )

        df = df.dropna()
        # split genres into multiple columns, so that each genre has its own column
        df = self.process_genres(
            df=df.copy(),
            item_id=self.args.item_id,
            genres_id=self.args.genres_id,
            threshold=getattr(
                self.args.data_processor, "genres_drop_threshold", 5
            ),
        )
        # get feature containing information if user watch latest movies
        df = self.get_years_since_release_viewed_feat(
            df=df.copy(),
            title_col=self.args.title_id,
            timestamp_col=self.args.date_id,
            col_name=getattr(
                self.args.data_processor, "duration_release_viewed_col", "NA"
            ),
        )
        # remove outliers and scale
        df = self.remove_outliers_and_scale(
            df,
            getattr(
                self.args.data_processor, "duration_release_viewed_col", "NA"
            ),
        )
        # split the data into train and test sets
        df = self.split_train_test(
            df=df.copy(),
            last_n_years=getattr(
                self.args.data_processor, "validation_data_duration", 1
            ),
            year_col=getattr(self.args.data_processor, "rating_year_col", "NA"),
        )
        # Register or update the dataset in Azure ML
        try:
            # log metrics
            self.log_metrics(df)
            # register or update the dataset in Azure ML
            register_or_update_dataset(df, "processed_item_ratings")
            # save the processed data to the data folder
            output_path = os.path.join(input_dir, "processed_data.pkl")
            logger.info(f"Saving data at {output_path}")
            df.to_pickle(output_path)
            # upload the processed data to Azure Blob Storage
            upload_data_frame_to_blob(
                account_url=getattr(
                    self.args.blob_params, "azure_account_url", "NA"
                ),
                container_name=getattr(
                    self.args.blob_params, "azure_container_name", "NA"
                ),
                file_name=getattr(
                    self.args.data_processor, "output_file", "NA"
                ),
                df=df,
            )

        except Exception as e:
            logger.error(
                f"Failed to register or update the dataset in Azure ML: {e}"
            )
        return df


if __name__ == "__main__":
    config = get_toml()
    # process data
    dp = DataProcessor.from_toml(config)
    df_processed = dp.process_data(df=pd.DataFrame())
