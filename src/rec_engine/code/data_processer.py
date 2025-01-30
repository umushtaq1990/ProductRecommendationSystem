from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from rec_engine.code.azure_utils import register_or_update_dataset
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
        df: pd.DataFrame, item_id: str, genres_col: str, threshold: int = 5
    ) -> pd.DataFrame:
        """
        Process the genres column in the DataFrame
        """
        logger.info("Processing genres")
        # Get unique items and genres
        df_items = df[[item_id, genres_col]].drop_duplicates()
        # Split genres into multiple columns, so that each genre has its own column
        df_genres = df_items[genres_col].str.get_dummies()
        # Rename (no genres listed) to "no_genres"
        df_genres = df_genres.rename(
            columns={"(no genres listed)": f"no_{genres_col}"}
        )
        # if genres columns contain less than 20 percent of records then drop the columns
        cols_to_drop = []
        for col in df_genres.columns:
            if (df_genres[col].sum() * 100 / df_genres.shape[0]) < threshold:
                cols_to_drop.append(col)
        df_genres = df_genres.drop(columns=cols_to_drop)
        logger.info(f"Columns dropped: {cols_to_drop}")
        # log the columns dropped
        df_items = df_items.join(df_genres)
        # Drop the original genres column
        df_items = df_items.drop(columns=[genres_col])
        # merge the processed genres back to the original dataframe
        df = df.merge(df_items, on=item_id)
        # Drop the original genres column
        df = df.drop(columns=[genres_col])
        logger.info("Genres processed successfully")
        return df_items

    @staticmethod
    def process_years(
        df: pd.DataFrame, title_col: str, timestamp_col: str
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
        df["years_since_release"] = df["year_watched"] - df["year_released"]
        # Drop records where years_since_release is missing
        df = df.dropna(subset=["years_since_release"])
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

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data
        """
        logger.info("Processing data")
        df = df.dropna()
        # split genres into multiple columns, so that each genre has its own column
        df = self.process_genres(
            df=df.copy(), item_id="movieId", genres_col="genres", threshold=5
        )
        df = self.process_years(
            df=df.copy(), title_col="title", timestamp_col="timestamp"
        )
        df = self.split_train_test(
            df=df.cpy(), last_n_years=2, year_col="year_watched"
        )
        # Register or update the dataset in Azure ML
        try:
            register_or_update_dataset(df, "processed_item_ratings")
        except Exception as e:
            logger.error(
                f"Failed to register or update the dataset in Azure ML: {e}"
            )
        return df


if __name__ == "__main__":
    config = get_toml()
    # load data
    from rec_engine.code.data_loader import DataLoader

    dl = DataLoader.from_toml(config)
    df = dl.load_data()
    # process data
    dp = DataProcessor.from_toml(config)
    df_processed = dp.process_data(df)
