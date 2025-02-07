import pandas as pd


def test_preprocess_data(
    raw_data: pd.DataFrame,
) -> None:
    """
    test data loading
    """
    df_res = raw_data.copy()
    # test if number of rows 14
    assert df_res.shape[0] == 14
    # test if number of columns 6
    assert df_res.shape[1] == 6
