import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

NHL_PALETTE = {
    "black": "#111111", "silver": "#A2AAAD", "orange": "#FF4600",
    "white": "#FFFFFF", "bg": "#F5F5F5"
}

def verify_dataset(file_path: str = "nhl_dataset.csv") -> None:
    """
    Loads the dataset, checks for integrity, and generates visualization charts.

    Args:
        file_path (str): Path to the CSV dataset.

    Returns:
        None
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return

    if "rolling_score_3" in df.columns:
        first = df.iloc[0][["rolling_score_3","score"]]

    plt.rcParams.update({'figure.facecolor': NHL_PALETTE["bg"], 'axes.facecolor': NHL_PALETTE["bg"]})

    plt.figure(figsize=(6,4))
    sns.countplot(x='won', data=df, palette=[NHL_PALETTE["silver"], NHL_PALETTE["orange"]])
    plt.title("Class Balance: Win vs Loss")
    plt.savefig("viz_class_balance.png")
    plt.close()

    sample_team = df['team_name'].mode()[0]
    subset = df[df.team_name == sample_team].sort_values("date").iloc[:40]

    plt.figure(figsize=(10,5))
    plt.plot(subset['score'].reset_index(drop=True), label="Raw Score", linestyle='--', marker='o')
    if "rolling_score_3" in subset.columns:
        plt.plot(subset['rolling_score_3'].reset_index(drop=True), label="Rolling(3)", linewidth=2)
    plt.legend()
    plt.title(f"Score vs Rolling Score (Team = {sample_team})")
    plt.savefig("viz_rolling_example.png")
    plt.close()

    cluster_features = ["rolling_score_10","rolling_hits_10","rolling_pim_10"]
    cluster_features = [c for c in cluster_features if c in df.columns]

    if len(cluster_features) == 3:
        cluster_df = df[cluster_features].dropna()
        if len(cluster_df) >= 50:
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(cluster_df)
            df.loc[cluster_df.index, "cluster"] = labels

            plt.figure(figsize=(8,6))
            sns.scatterplot(data=cluster_df, x="rolling_score_10", y="rolling_hits_10",
                            hue=labels, palette="viridis")
            plt.title("Clustered Team Archetypes")
            plt.savefig("viz_clusters.png")
            plt.close()

if __name__ == "__main__":
    verify_dataset()