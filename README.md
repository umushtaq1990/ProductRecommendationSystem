# Product Recommendation System
This Python project is a powerful product-based recommendation engine designed to boost customer engagement and drive business growth. By analyzing user behavior, product features, and purchase history, it delivers personalized product suggestions. 


This repository contains a recommendation engine developed using two techniques:

- **Singular Value Decomposition (SVD)** - A matrix factorization method for extracting latent features.
- **Collaborative Filtering (CF)** - A popular recommendation technique based on user-item interactions.

The goal of this recommendation system is to predict products a user may be interested in by analyzing past behavior and similar users' interactions with products.

### Key Features
- **Product Recommendations**: Suggests products based on user preferences.
- **SVD and CF Algorithms**: Implements both matrix factorization (SVD) and collaborative filtering techniques for generating recommendations.
- **Model Evaluation**: Uses metrics like recall to evaluate the performance of the recommendation system.

### Workflow

The following diagram represents the workflow of the recommender system:

![Workflow Diagram](documentation/workflow.svg)

Before going into explaination of this workflow lets fetch the git repository containing code and prepare the enviornment.

### Installation

#### Prerequisites
Make sure you have Conda installed. If you don’t have it yet, you can download it from [Anaconda's website](https://www.anaconda.com/products/distribution) or install it via Miniconda.


**Step 1: Create a Conda Environment**

To set up a clean environment with all necessary dependencies, you can create a new Conda environment.

Run the following commands in your terminal or command prompt:

```bash
conda create -n product-recommendation python=3.8
conda activate product-recommendation
```

This will create a new Conda environment named `product-recommendation` with Python 3.8 installed and activate the environment.

**Step 2: Install Dependencies**

Once the environment is activated, install the required dependencies from environment.yml.

```bash
conda env create -f environment.yml
```
**Step 3: Clone the Repository**

To get started, clone this repository:

```bash
git clone https://github.com/umushtaq1990/product-recommendation-engine.git

cd product-recommendation-engine
```

Now, your environment is set up and ready for use.


### Project Directory Structure

Below is the directory structure for the recommender system project:

📂 recommender-system  

    │── 📄 README.md # Project documentation  

    │── 📂 data # Dataset folder  
        ├── 📄 interactions.csv # User interactions data  
        ├── 📄 items.csv # Items metadata   

    │── 📂 environment # contain files to set enviornment  
        ├── 📄 environment.yml # yaml file containing required packages   

    │── 📂 src # Source code folder  
        ├── 📄 main.py # main script
        │── 📂 rec_engine # Source code folder  
            ├── 📄 config.toml # configuration file
            ├── 📄 version.py # contains package version 
            │── 📂 code # Source code folder 
                ├── 📄 data_loader.py # handles data loading/fetching  
                ├── 📄 data_processor.py # used for data cleaning
                ├── 📄 model.py # contain ml algorithims  
                ├── 📄 evaluation.py # evaluate prediction results
                ├── 📄 config.py # handles config file 
                ├── 📄 pipeline.py # integrate modules
        │── 📂 tests # Unit tests  
            │── 📂 .test_data # folder containing sample test datsets
            ├── 📄 conftest.py # pytest configuration file
            ├── 📄 test_data_processor.py # contain unit tests for data processer module
            ├── 📄 test_modeling.py # contain unit tests for modling module
            ├── 📄 test_evaluator.py # contain unit tests for evaluation module

    │── 📂  documentation # Diagrams and visualizations  
        ├── 📄 workflow.svg # Workflow diagram  

    │── 📂 deployment # Deployment-related files  
        ├── 📄 Dockerfile # Docker configuration  
        ├── 📄 app.py # API endpoint  
        ├── 📄 config.yaml # Configuration file  


### Contributing
If you have any suggestions or improvements for this project, feel free to fork the repository and submit a pull request.

#### Steps to contribute:
- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Make your changes and commit (git commit -am 'Add feature').
- Push the branch (git push origin feature/your-feature).
- Create a pull request.


e.g: If evaluaton data
contain 100 users,
those 100 users are
checked for
recommendations
generated
by different models 
by taking into account
top n recommended
results.If 70 users
newly taken product matches with
recommended results, recall
score for that model
will be 0.7 






