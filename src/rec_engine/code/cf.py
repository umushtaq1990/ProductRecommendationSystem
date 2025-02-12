import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("HybridRecommendation")


class HybridRecommendation:
    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "cosine",
        user_features: Optional[List[str]] = None,
        item_features: Optional[List[str]] = None,
    ):
        """
        Initialize the Hybrid Recommendation model.

        :param n_neighbors: Number of neighbors to use for k-NN.
        :param metric: Metric to use for computing similarity (default: cosine).
        :param user_features: List of user feature columns to use.
        :param item_features: List of item feature columns to use.
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.user_features = user_features
        self.item_features = item_features
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.model: Optional[NearestNeighbors] = None
        self.user_feature_transformer: Optional[ColumnTransformer] = None
        self.item_feature_transformer: Optional[ColumnTransformer] = None
        self.user_features_transformed: Optional[np.ndarray] = None
        self.item_features_transformed: Optional[np.ndarray] = None

    def fit(
        self, ratings: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame
    ) -> None:
        """
        Fit the hybrid recommendation model to the data.

        :param ratings: DataFrame containing user_id, item_id, and rating columns.
        :param users: DataFrame containing user_id and additional user features.
        :param items: DataFrame containing item_id and additional item features.
        """
        self._validate_data(ratings, users, items)

        # if self.user_features is None or self.item_features is None raise error
        if self.user_features is None or self.item_features is None:
            raise ValueError(
                "User and item features must be provided for the hybrid model."
            )

        # Preprocess user and item features
        self.user_feature_transformer = self._create_feature_transformer(
            users, self.user_features
        )
        self.item_feature_transformer = self._create_feature_transformer(
            items, self.item_features
        )

        self.user_features_transformed = (
            self.user_feature_transformer.fit_transform(users)
        )
        self.item_features_transformed = (
            self.item_feature_transformer.fit_transform(items)
        )

        # Create user-item interaction matrix
        self.user_item_matrix = ratings.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0
        )

        # Combine user interaction history with user features
        user_feature_vectors = self._create_user_feature_vectors(
            self.user_item_matrix, self.user_features_transformed
        )

        # Fit the k-NN model on user-level feature vectors
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self.metric
        )
        self.model.fit(user_feature_vectors)
        logger.info("Hybrid Recommendation model fitted successfully")

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

        # Get the user's feature vector
        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_feature_vector = self._create_user_feature_vectors(
            self.user_item_matrix, self.user_features_transformed
        )[user_index].reshape(1, -1)

        # Find the k-nearest neighbors of the user
        distances, indices = self.model.kneighbors(user_feature_vector)

        # Get the ratings of the nearest neighbors for the item
        neighbor_ratings = self.user_item_matrix.iloc[indices.flatten()][
            item_id
        ]

        # Predict the rating as the average of the neighbors' ratings
        predicted_rating = float(neighbor_ratings.mean())
        return predicted_rating

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

        # Get unrated items for the user
        user_index = self.user_item_matrix.index.get_loc(user_id)
        unrated_items = self.user_item_matrix.columns[
            self.user_item_matrix.iloc[user_index] == 0
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

    def _create_feature_transformer(
        self, data: pd.DataFrame, features: List[str]
    ) -> ColumnTransformer:
        """
        Create a feature transformer for user or item features.

        :param data: DataFrame containing the features.
        :param features: List of feature columns to use.
        :return: ColumnTransformer for preprocessing features.
        """
        numeric_features = (
            data[features].select_dtypes(include=np.number).columns
        )
        categorical_features = (
            data[features].select_dtypes(exclude=np.number).columns
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )
        return transformer

    def _create_user_feature_vectors(
        self, user_item_matrix: pd.DataFrame, user_features: np.ndarray
    ) -> np.ndarray:
        """
        Create user-level feature vectors by combining interaction history and user features.

        :param user_item_matrix: User-item interaction matrix.
        :param user_features: Transformed user features.
        :return: Combined user feature vectors.
        """
        # Combine interaction history and user features
        user_feature_vectors = np.hstack(
            [user_item_matrix.values, user_features]
        )
        return user_feature_vectors

    def _combine_features(
        self,
        user_item_matrix: pd.DataFrame,
        user_features: np.ndarray,
        item_features: np.ndarray,
    ) -> np.ndarray:
        """
        Combine user-item interaction matrix with user and item features.

        :param user_item_matrix: User-item interaction matrix.
        :param user_features: Transformed user features.
        :param item_features: Transformed item features.
        :return: Combined feature matrix.
        """
        n_users, n_items = user_item_matrix.shape
        n_user_features = user_features.shape[1]
        n_item_features = item_features.shape[1]

        # Create a combined feature matrix for each user-item pair
        combined_features = []

        for user_idx in range(n_users):
            for item_idx in range(n_items):
                # Get the user and item features
                user_feature = user_features[user_idx]
                item_feature = item_features[item_idx]

                # Get the interaction value (rating)
                interaction_value = user_item_matrix.iloc[user_idx, item_idx]

                # Combine all features into a single vector
                combined_vector = np.hstack(
                    [[interaction_value], user_feature, item_feature]
                )
                combined_features.append(combined_vector)

        return np.array(combined_features)

    def _validate_data(
        self, ratings: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame
    ) -> None:
        """
        Validate the input data to ensure it contains the required columns.

        :param ratings: DataFrame containing user_id, item_id, and rating columns.
        :param users: DataFrame containing user_id and additional user features.
        :param items: DataFrame containing item_id and additional item features.
        """
        required_ratings_columns = {"user_id", "item_id", "rating"}
        if not required_ratings_columns.issubset(ratings.columns):
            raise ValueError(
                f"Ratings DataFrame must contain the following columns: {required_ratings_columns}"
            )

        if "user_id" not in users.columns:
            raise ValueError("Users DataFrame must contain a 'user_id' column.")

        if "item_id" not in items.columns:
            raise ValueError(
                "Items DataFrame must contain an 'item_id' column."
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
    users_data = {
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 40, 26, 38, 27],
        "location": ["A", "B", "A", "B", "A"],
    }
    items_data = {
        "item_id": [1, 2, 3, 4, 5],
        "genre": ["Action", "Animation", "Action", "Comedy", "Animation"],
    }

    ratings_df = pd.DataFrame(data)
    # drop timestamp column
    ratings_df = ratings_df.drop(columns=["timestamp"])
    users_df = pd.DataFrame(users_data)
    items_df = pd.DataFrame(items_data)

    # Initialize and fit the model
    hybrid_model = HybridRecommendation(
        n_neighbors=3,
        user_features=["age", "location"],
        item_features=["genre"],
    )
    hybrid_model.fit(ratings_df, users_df, items_df)

    # Generate recommendations for a user
    user_id = 1
    recommendations = hybrid_model.recommend(user_id, n_recommendations=2)
    print(f"Recommendations for User {user_id}:")
    print(recommendations)
    print
