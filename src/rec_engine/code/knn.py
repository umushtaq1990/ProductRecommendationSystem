import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("CollaborativeFiltering")


class CollaborativeFiltering:
    def __init__(self, n_neighbors: int = 5, metric: str = "cosine"):
        """
        Initialize the Collaborative Filtering model.

        :param n_neighbors: Number of neighbors to use for k-NN.
        :param metric: Metric to use for computing similarity (default: cosine).
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.model: Optional[NearestNeighbors] = None
        self.global_mean: Optional[float] = None
        self.user_bias: Optional[pd.Series] = None
        self.item_bias: Optional[pd.Series] = None

    def fit(self, ratings: pd.DataFrame) -> None:
        """
        Fit the collaborative filtering model to the ratings data.

        :param ratings: DataFrame containing user_id, item_id, and rating columns.
        """
        self._validate_ratings(ratings)
        self.user_item_matrix = ratings.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0
        )
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self.metric
        )
        self.model.fit(self.user_item_matrix)

        # Calculate global mean, user bias, and item bias
        self.global_mean = ratings["rating"].mean()
        self.user_bias = (
            ratings.groupby("user_id")["rating"].mean() - self.global_mean
        )
        self.item_bias = (
            ratings.groupby("item_id")["rating"].mean() - self.global_mean
        )

        logger.info("Collaborative Filtering model fitted successfully")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating for a given user-item pair.

        :param user_id: The user ID.
        :param item_id: The item ID.
        :return: Predicted rating.
        """
        if self.user_item_matrix is None or self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Please call the fit method first."
            )

        if user_id not in self.user_item_matrix.index:
            raise ValueError(
                f"User ID {user_id} not found in the training data."
            )

        if item_id not in self.user_item_matrix.columns:
            raise ValueError(
                f"Item ID {item_id} not found in the training data."
            )

        # Find the k-nearest neighbors of the user
        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_item_matrix.iloc[user_index].values.reshape(
            1, -1
        )
        distances, indices = self.model.kneighbors(user_vector)

        # Get the ratings of the nearest neighbors for the item
        neighbor_ratings = self.user_item_matrix.iloc[indices.flatten()][
            item_id
        ]
        # Exclude neighbors who have not rated the item
        neighbor_ratings = neighbor_ratings[neighbor_ratings > 0]

        # Predict the rating as the average of the neighbors' ratings
        if not neighbor_ratings.empty:
            predicted_rating = neighbor_ratings.mean()
        else:
            predicted_rating = self.global_mean

        # Adjust for user and item bias
        # make sure self.user_bias and self.item_bias are not None
        if self.user_bias is None or self.item_bias is None:
            raise ValueError("User bias and item bias are not calculated.")
        user_bias = self.user_bias.get(user_id, 0)
        item_bias = self.item_bias.get(item_id, 0)
        predicted_rating += user_bias + item_bias

        return float(predicted_rating)

    def recommend(
        self, user_id: int, n_recommendations: int = 5
    ) -> pd.DataFrame:
        """
        Generate recommendations for a given user.

        :param user_id: The user ID.
        :param n_recommendations: Number of recommendations to generate.
        :return: DataFrame containing item_id and predicted_rating for recommended items.
        """
        if self.user_item_matrix is None or self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Please call the fit method first."
            )

        if user_id not in self.user_item_matrix.index:
            raise ValueError(
                f"User ID {user_id} not found in the training data."
            )

        # Find the k-nearest neighbors of the user
        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_item_matrix.iloc[user_index].values.reshape(
            1, -1
        )
        distances, indices = self.model.kneighbors(user_vector)

        # Get the items rated by the neighbors but not by the user
        user_rated_items = set(
            self.user_item_matrix.columns[
                self.user_item_matrix.iloc[user_index] > 0
            ]
        )
        neighbor_ratings = self.user_item_matrix.iloc[indices.flatten()]
        unrated_items = neighbor_ratings.columns[
            ~neighbor_ratings.columns.isin(user_rated_items)
        ]

        # Predict ratings for the unrated items
        recommendations = []
        for item_id in unrated_items:
            predicted_rating = self.predict(user_id, item_id)
            recommendations.append((item_id, predicted_rating))

        # Sort by predicted rating and return the top N recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(
            recommendations[:n_recommendations],
            columns=["item_id", "predicted_rating"],
        )

    def get_similar_users(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        """
        Get the top N similar users to the given user_id.

        :param user_id: The user ID.
        :param top_n: Number of similar users to return.
        :return: DataFrame containing similar_user_id and similarity scores.
        """
        if self.user_item_matrix is None or self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Please call the fit method first."
            )

        if user_id not in self.user_item_matrix.index:
            raise ValueError(
                f"User ID {user_id} not found in the training data."
            )

        # Find the k-nearest neighbors of the user
        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_item_matrix.iloc[user_index].values.reshape(
            1, -1
        )
        distances, indices = self.model.kneighbors(user_vector)

        # Get the similarity scores and user IDs of the neighbors
        similarities = 1 - distances.flatten()
        similar_users = self.user_item_matrix.index[indices.flatten()]

        # Create a DataFrame with the results
        return pd.DataFrame(
            {
                "similar_user_id": similar_users,
                "similarity": similarities,
            }
        ).head(top_n)

    def get_all_user_similarities(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get the top N similar users for all users in the dataset.

        :param top_n: Number of similar users to return for each user.
        :return: DataFrame containing user_id, similar_user_id, and similarity scores.
        """
        if self.user_item_matrix is None or self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Please call the fit method first."
            )

        all_similarities = []

        for user_id in self.user_item_matrix.index:
            similar_users = self.get_similar_users(user_id, top_n)
            similar_users["user_id"] = user_id
            all_similarities.append(similar_users)

        return pd.concat(all_similarities, ignore_index=True)

    def get_all_user_recommendations(
        self, n_recommendations: int = 5
    ) -> pd.DataFrame:
        """
        Get recommendations for all users in the dataset.

        :param n_recommendations: Number of recommendations to generate for each user.
        :return: DataFrame containing user_id, item_id, and predicted_rating for recommended items.
        """
        if self.user_item_matrix is None or self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Please call the fit method first."
            )

        all_recommendations = []

        for user_id in self.user_item_matrix.index:
            recommendations = self.recommend(user_id, n_recommendations)
            recommendations["user_id"] = user_id
            all_recommendations.append(recommendations)

        return pd.concat(all_recommendations, ignore_index=True)

    def _validate_ratings(self, ratings: pd.DataFrame) -> None:
        """
        Validate the ratings DataFrame to ensure it contains the required columns.

        :param ratings: DataFrame containing user_id, item_id, and rating columns.
        """
        required_columns = {"user_id", "item_id", "rating"}
        if not required_columns.issubset(ratings.columns):
            raise ValueError(
                f"Ratings DataFrame must contain the following columns: {required_columns}"
            )


# Example Usage
if __name__ == "__main__":
    # Sample data
    data = {
        "user_id": [
            1,
            1,
            1,
            1,  # user close to 2
            2,
            2,
            2,
            2,
            3,
            3,
            4,
            4,
            4,  # user close to 1
            5,
            5,
            5,
        ],  # user close to 2
        "item_id": [1, 2, 3, 4, 1, 2, 3, 5, 2, 3, 1, 3, 4, 1, 2, 5],
        "rating": [5, 4, 5, 4, 5, 4, 5, 5, 4, 5, 3, 4, 3, 4, 3, 4],
        "timestamp": [
            1112484027,
            1112484676,
            1112484819,
            1112484919,
            1112484580,
            1112484727,
            1112484927,
            1112484955,
            1112484580,
            1112484727,
            1112484680,
            1112484780,
            1112484880,
            1112484580,
            1112484700,
            1112484900,
        ],
    }
    ratings_df = pd.DataFrame(data)

    # get pivot table
    ratings_df_pivot = ratings_df.pivot_table(
        index="user_id", columns="item_id", values="rating"
    ).reset_index()

    print("Data:")
    print(ratings_df_pivot)

    # Initialize and fit the model
    cf_model = CollaborativeFiltering(n_neighbors=4)
    cf_model.fit(ratings_df)

    # Generate recommendations for a user
    user_id = 1
    recommendations = cf_model.recommend(user_id, n_recommendations=2)
    print(f"Recommendations for User {user_id}:")
    print(recommendations)

    # Get recommendations for all users
    all_recommendations = cf_model.get_all_user_recommendations(
        n_recommendations=2
    )
    # get pivot table
    recommendations_pivot = all_recommendations.pivot_table(
        index="user_id", columns="item_id", values="predicted_rating"
    ).reset_index()
    print("Recommendations:")
    print(recommendations_pivot)

    # Get similar users for a user
    similar_users = cf_model.get_similar_users(user_id, top_n=2)
    print(f"\nSimilar Users for User {user_id}:")
    print(similar_users)
    print("\n")

    # Get similar users for all users
    all_similarities = cf_model.get_all_user_similarities(top_n=5)
    # Create a pivot table for the heatmap
    pivot_table = all_similarities.pivot(
        index="user_id", columns="similar_user_id", values="similarity"
    )
    print(pivot_table)
    print("\n")
