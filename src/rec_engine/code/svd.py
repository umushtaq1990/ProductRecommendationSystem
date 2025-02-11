import logging
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from surprise import SVD, Dataset, Reader, Trainset, accuracy
from surprise.model_selection import train_test_split

logger = logging.getLogger("SVDModel")


class SVDModel:
    def __init__(
        self, n_factors: int = 15, n_epochs: int = 20, random_state: int = 42
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
        )
        self.trainset: Optional[Trainset] = None

    def fit(self, ratings: pd.DataFrame) -> None:
        """
        Fit the SVD model to the ratings data.

        :param ratings: DataFrame containing user_id, item_id, rating, and timestamp columns
        """
        self._validate_ratings(ratings)
        reader = Reader(
            rating_scale=(ratings["rating"].min(), ratings["rating"].max())
        )
        data = Dataset.load_from_df(
            ratings[["user_id", "item_id", "rating"]], reader
        )
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)
        logger.info("SVD model fitted successfully")

    def fit_predict(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the SVD model to the ratings data and return predictions.

        :param ratings: DataFrame containing user_id, item_id, rating, and timestamp columns
        :return: DataFrame containing user_id, item_id, and predicted rating
        """
        self.fit(ratings)
        return self.predict(ratings)

    def predict(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Predict ratings for the given user-item pairs.

        :param ratings: DataFrame containing user_id, item_id, rating, and timestamp columns
        :return: DataFrame containing user_id, item_id, and predicted rating
        """
        self._validate_ratings(ratings)
        predictions = []
        for _, row in ratings.iterrows():
            pred = self.model.predict(row["user_id"], row["item_id"])
            predictions.append((row["user_id"], row["item_id"], pred.est))
        return pd.DataFrame(
            predictions, columns=["user_id", "item_id", "predicted_rating"]
        )

    def recommend(
        self,
        ratings: pd.DataFrame,
        n_recommendations: int = 5,
        use_timestamp: bool = False,
    ) -> pd.DataFrame:
        """
        Generate recommendations for each user.

        :param ratings: DataFrame containing user_id, item_id, rating, and timestamp columns
        :param n_recommendations: Number of recommendations to generate for each user
        :param use_timestamp: Boolean flag to turn on/off the timestamp effect
        :return: DataFrame containing user_id, item_id, and predicted rating for recommended items
        """
        self._validate_ratings(ratings)
        unique_users = ratings["user_id"].unique()
        unique_items = ratings["item_id"].unique()
        recommendations = []

        if use_timestamp:
            # Normalize timestamps to a range between 0 and 1
            min_timestamp = ratings["timestamp"].min()
            max_timestamp = ratings["timestamp"].max()
            ratings["normalized_timestamp"] = (
                ratings["timestamp"] - min_timestamp
            ) / (max_timestamp - min_timestamp)

        for user_id in unique_users:
            user_ratings = ratings[ratings["user_id"] == user_id]
            rated_items = set(user_ratings["item_id"])
            unrated_items = [
                item for item in unique_items if item not in rated_items
            ]

            user_recommendations = []
            for item_id in unrated_items:
                pred = self.model.predict(user_id, item_id)
                if use_timestamp:
                    # Apply exponential decay to the predicted rating based on the normalized timestamp
                    latest_timestamp = user_ratings[
                        "normalized_timestamp"
                    ].max()
                    weight = np.exp(
                        latest_timestamp
                        - user_ratings["normalized_timestamp"].mean()
                    )
                    weighted_pred = pred.est * weight
                else:
                    weighted_pred = pred.est
                user_recommendations.append((user_id, item_id, weighted_pred))

            user_recommendations.sort(key=lambda x: x[2], reverse=True)
            recommendations.extend(user_recommendations[:n_recommendations])

        return pd.DataFrame(
            recommendations, columns=["user_id", "item_id", "predicted_rating"]
        )

    def get_similar_users(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        """
        Get the top N similar users to the given user_id based on latent factors.

        :param user_id: The user_id for which to find similar users
        :param top_n: The number of similar users to return
        :return: DataFrame containing similar user_ids and their similarity scores
        """
        if self.trainset is None:
            raise ValueError(
                "Model has not been fitted yet. Please call the fit method first."
            )

        user_inner_id = self.trainset.to_inner_uid(user_id)
        user_factors = self.model.pu[user_inner_id]

        similarities = []
        for other_user_inner_id in range(self.trainset.n_users):
            if other_user_inner_id != user_inner_id:
                other_user_factors = self.model.pu[other_user_inner_id]
                similarity = np.dot(user_factors, other_user_factors) / (
                    np.linalg.norm(user_factors)
                    * np.linalg.norm(other_user_factors)
                )
                other_user_id = self.trainset.to_raw_uid(other_user_inner_id)
                similarities.append((other_user_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(
            similarities[:top_n], columns=["similar_user_id", "similarity"]
        )

    def get_all_similar_users(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get the top N similar users for all users based on latent factors.

        :param top_n: The number of similar users to return for each user
        :return: DataFrame containing user_id, similar_user_id, and similarity scores
        """
        if self.trainset is None:
            raise ValueError(
                "Model has not been fitted yet. Please call the fit method first."
            )

        all_similar_users = []
        for user_id in self.trainset.all_users():
            raw_user_id = self.trainset.to_raw_uid(user_id)
            similar_users = self.get_similar_users(raw_user_id, top_n)
            similar_users["user_id"] = raw_user_id
            all_similar_users.append(similar_users)

        return pd.concat(all_similar_users, ignore_index=True)

    def _validate_ratings(self, ratings: pd.DataFrame) -> None:
        """
        Validate the ratings DataFrame to ensure it contains the required columns.

        :param ratings: DataFrame containing user_id, item_id, rating, and timestamp columns
        """
        required_columns = {"user_id", "item_id", "rating", "timestamp"}
        if not required_columns.issubset(ratings.columns):
            raise ValueError(
                f"Ratings DataFrame must contain the following columns: {required_columns}"
            )


def save_heatmap(pivot_table: pd.DataFrame, dir_path: str) -> None:
    """
    Save the heatmap of user similarities.

    :param pivot_table: Pivot table containing user similarities
    :param dir_path: Directory path to save the heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", center=0)
    plt.title("User Similarity Heatmap")
    plt.savefig(f"{dir_path}/user_similarity_heatmap.png")
    plt.show()


def save_rmse_plot(rmse_values: List[Tuple[int, float]], dir_path: str) -> None:
    """
    Save the plot of RMSE vs. n_factors.

    :param rmse_values: List of tuples containing n_factors and corresponding RMSE values
    :param dir_path: Directory path to save the plot
    """
    n_factors_list, rmse_list = zip(*rmse_values)
    plt.figure(figsize=(10, 6))
    plt.plot(n_factors_list, rmse_list, marker="o")
    plt.xlabel("Number of Latent Factors (n_factors)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Number of Latent Factors")
    plt.grid(True)
    plt.savefig(f"{dir_path}/rmse_vs_n_factors.png")
    plt.show()


def find_optimal_n_factors(
    ratings_df: pd.DataFrame, test_size: float = 0.2, dir_path: str = "data"
) -> int:
    """
    Find the optimal number of latent factors based on RMSE.

    :param ratings_df: DataFrame containing user_id, item_id, rating, and timestamp columns
    :param test_size: Proportion of the dataset to include in the test split
    :param dir_path: Directory path to save the RMSE plot
    :return: Optimal number of latent factors
    """

    num_users = ratings_df["user_id"].nunique()
    num_items = ratings_df["item_id"].nunique()

    # Generate a dynamic range of n_factors based on the number of users and items
    max_factors = min(num_users, num_items)
    n_factors_list = np.unique(
        np.logspace(0, np.log10(max_factors), num=10, dtype=int)
    )

    best_n_factors = int(n_factors_list[0])
    best_rmse = float("inf")
    rmse_values = []

    for n_factors in n_factors_list:
        svd_model = SVDModel(n_factors=n_factors)
        trainset, testset = train_test_split(
            Dataset.load_from_df(
                ratings_df[["user_id", "item_id", "rating"]],
                Reader(rating_scale=(1, 5)),
            ),
            test_size=test_size,
        )
        svd_model.model.fit(trainset)
        predictions = svd_model.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        rmse_values.append((n_factors, rmse))
        print(f"n_factors: {n_factors}, RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_n_factors = n_factors

    # Save the RMSE plot
    save_rmse_plot(rmse_values, dir_path)

    print(f"Optimal n_factors: {best_n_factors}, RMSE: {best_rmse}")
    return best_n_factors


if __name__ == "__main__":
    # Example usage
    data = {
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
        "item_id": [1, 2, 3, 1, 2, 3, 2, 3],
        "rating": [5, 4, 5, 5, 4, 5, 4, 5],
        "timestamp": [
            1112484027,
            1112484676,
            1112484819,
            1112484580,
            1112484727,
            1112484927,
            1112484580,
            1112484727,
        ],
    }
    ratings_df = pd.DataFrame(data)
    # get pivot table
    ratings_df_pivot = ratings_df.pivot_table(
        index="user_id", columns="item_id", values="rating"
    ).reset_index()

    print("Data:")
    print(ratings_df_pivot)
    ratings_df["date"] = ratings_df["timestamp"].apply(
        lambda x: datetime.fromtimestamp(x).strftime("%H%M%S")
    )
    date_pivot = ratings_df.pivot_table(
        index="user_id", columns="item_id", values="date"
    ).reset_index()
    print(date_pivot)

    # Find the optimal number of latent factors
    optimal_n_factors = find_optimal_n_factors(ratings_df)

    # Fit the SVD model with the optimal number of latent factors
    svd_model = SVDModel(n_factors=optimal_n_factors)
    svd_model.fit(ratings_df)
    predictions = svd_model.fit_predict(ratings_df)
    recommendations = svd_model.recommend(ratings_df)

    # get pivot table
    recommendations_pivot = recommendations.pivot_table(
        index="user_id", columns="item_id", values="predicted_rating"
    ).reset_index()
    print("Data:")
    print(recommendations_pivot)

    # Get similar users for all users
    all_similar_users = svd_model.get_all_similar_users(top_n=3)
    print("\nAll Similar Users:")
    print(all_similar_users)

    # Create a pivot table for the heatmap
    pivot_table = all_similar_users.pivot(
        index="user_id", columns="similar_user_id", values="similarity"
    )

    # Plot and save the heatmap
    save_heatmap(
        pivot_table, dir_path="data"
    )  # Change this to your desired directory
    print("Done")
