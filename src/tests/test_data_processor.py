import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Add the src directory to the sys.path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

from rec_engine.code.data_processor import DataProcessor


def test_process_genres(
    raw_data: pd.DataFrame,
    data_processor_module: DataProcessor,
    config: Dict[str, Any],
) -> None:
    """Test the process_genres function."""
    # Call the function
    result = data_processor_module.process_genres(
        df=raw_data,
        item_id=config["args"]["item_id"],
        genres_id=config["args"]["genres_id"],
        threshold=config["args"]["data_processor"]["genres_drop_threshold"],
    )

    # Check if the original 'genres' column is dropped
    assert "genres" not in result.columns

    # Check if the new genre columns are created
    expected_genres = [
        "Children",
        "Adventure",
        "Comedy",
        "Romance",
        "Animation",
        "Fantasy",
        "Drama",
    ]
    for genre in expected_genres:
        assert genre in result.columns

    # Check if the 'no_genres' column is not there as all items contain atleat one genre
    assert "no_genres" not in result.columns

    # Check if the threshold logic works
    # For example, if a genre appears in less than 5% of items, it should be dropped
    assert (
        len(result.columns) == len(expected_genres) + 5
    )  # +5 for 'item_id', 'user_id', 'date', 'rating', 'title'


def test_threshold_logic(
    raw_data: pd.DataFrame,
    data_processor_module: DataProcessor,
    config: Dict[str, Any],
) -> None:
    """Test the threshold logic for dropping genres."""
    # Call the function with a higher threshold (e.g., 50%)
    result = data_processor_module.process_genres(
        df=raw_data,
        item_id=config["args"]["item_id"],
        genres_id=config["args"]["genres_id"],
        threshold=50,
    )

    # Check if genres appearing in less than 50% of items are dropped
    assert "Action" not in result.columns
    assert "Adventure" not in result.columns
    assert "Romance" not in result.columns
    assert "no_genres" not in result.columns
    # Check if genres appearing in more than 50% of items are kept
    assert "Comedy" in result.columns
