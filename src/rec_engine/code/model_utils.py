from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def get_similar_users_cosine(
    ratings_df: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    pivot_table = ratings_df.pivot_table(
        index="user_id", columns="item_id", values="rating"
    ).fillna(0)
    similarity_matrix = cosine_similarity(pivot_table)
    similarity_df = pd.DataFrame(
        similarity_matrix, index=pivot_table.index, columns=pivot_table.index
    )

    all_similar_users = []
    for user_id in similarity_df.index:
        similar_users = (
            similarity_df[user_id].drop(user_id).nlargest(top_n).reset_index()
        )
        similar_users.columns = ["similar_user_id", "similarity"]
        similar_users["user_id"] = user_id
        all_similar_users.append(similar_users)

    return pd.concat(all_similar_users, ignore_index=True)


def save_heatmap(
    pivot_table: pd.DataFrame, dir_path: str, file_name: str
) -> None:
    """
    Save the heatmap of user similarities.

    :param pivot_table: Pivot table containing user similarities
    :param dir_path: Directory path to save the heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", center=0)
    plt.title("User Similarity Heatmap")
    plt.savefig(f"{dir_path}/{file_name}")


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
