# DataLoader Module

The `DataLoader` module is responsible for loading data from various sources, including Azure ML registered data components, Azure Blob Storage, and local files. This document provides an overview of the `DataLoader` class and its methods, along with a workflow diagram.

## Setting Parameters in the Configuration File

The `DataLoader` module relies on a [`configuration file`](../src/rec_engine/config.toml), typically in TOML format, to set various parameters required for loading data. The configuration file should include following paremeters in ```args``` and ```args.dataloader``` sections to work properly.

```toml
[args]
data_folder = "data" # local folder where raw data files are available

[args.data_loader]
azure_account_url = "https://your_account_url"  # to access blob_storage
azure_container_name = "your_container_name" # directory on blob storage
item_file = "movies.csv" # items file name
rating_file = "ratings.csv" # user-rating file name
az_ws_item_rating_data = "item_rating_data" # azure ml workspace containing registered data
```

## How the Logic Works
### Initialization
- The ```DataLoader``` class is initialized with a configuration dictionary. This dictionary can be loaded from a TOML file using the from_toml class method.
- The ```ParametersConfig``` class processes the configuration dictionary to extract relevant parameters.

### Loading Data
The ```load_data``` method is the main entry point for loading data. It attempts to load data from various sources in a specific order:

- Azure ML Registered Data Component:
The method first checks if the data is available in an Azure ML registered data component. If available, it loads the data using the ```load_data_from_azure_ml``` method.

- Azure Blob Storage:
If the data is not available in Azure ML, the method attempts to load data from Azure Blob Storage using the ```load_data_from_blob``` method. This method downloads the item and rating files from the specified Azure Blob Storage container.

- Local Directory:
If the data is not available in Azure Blob Storage, the method falls back to loading data from a local directory using the ```load_data_from_local``` method. This method reads the item and rating files from the specified local directory.

### Uploading Data
- If the data is loaded from the local directory, the ```upload_data_to_blob``` method is called to upload the item and rating files to Azure Blob Storage. This ensures that the data is available in Azure Blob Storage for future use.

### Merging Data
- After loading the data, the item and rating DataFrames are merged based on the ```item_id``` parameter. This merged DataFrame contains the combined information from both files.

### Saving Data
- If the data was not initially available in the Azure ML registered data component, the merged DataFrame is saved to Azure ML using the ```Dataset.Tabular.register_pandas_dataframe method```. This makes the data available as a registered dataset in Azure ML for future use.

By following this logic, the ```DataLoader``` module ensures that data is loaded efficiently from the most appropriate source and is made available for further processing and analysis.

### Workflow diagram
The following diagram represents the data loading workflow:
![Workflow Diagram](data_loader_workflow.svg)
