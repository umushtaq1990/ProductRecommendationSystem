from rec_engine.code.data_loader import DataLoader


def test_preprocess_data(
    data_loader_module: DataLoader,
) -> None:
    """
    test data loading
    """
    df_res = data_loader_module.load_data()
    # test if number of rows 14
    assert df_res.shape[0] == 14
    # test if number of columns 6
    assert df_res.shape[1] == 6
