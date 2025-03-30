#%%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# クラスタ内の統計量を計算する関数
def cluster_cohesion_stats_by_cluster(embeddings, labels):
    unique_labels = np.unique(labels)
    cluster_stats = {}

    for label in unique_labels:
        # クラスタ内のポイントを取得
        cluster_points = embeddings[labels == label]
        n_samples = len(cluster_points)

        # サンプル数が1以下の場合は初期値をセット
        if n_samples > 1:
            similarities = cosine_similarity(cluster_points)
            upper_indices = np.triu_indices(n_samples, k=1)
            similarity_values = similarities[upper_indices]

            stats = {
                'count': n_samples,  # サンプル数の追加
                'mean': np.mean(similarity_values),
                'std': np.std(similarity_values),
                'var': np.var(similarity_values),
                'max': np.max(similarity_values),
                'min': np.min(similarity_values),
                'median': np.median(similarity_values),
            }
        else:
            # サンプルが1つ以下の場合、類似度計算は不要
            stats = {
                'count': n_samples,  # サンプル数のみ
                'mean': 0,
                'std': 0,
                'var': 0,
                'max': 0,
                'min': 0,
                'median': 0,
            }

        # クラスタごとの統計量を保存
        cluster_stats[label] = stats

    return cluster_stats

# サンプルデータ生成
np.random.seed(42)
embeddings = np.random.rand(10, 5)  # 10個のサンプル、5次元の埋め込みベクトル
labels = np.array([0, 1, 0, 1, 0, 2, 2, 1, 0, 2])  # 3つのクラスタ

# 関数の動作確認
cluster_stats = cluster_cohesion_stats_by_cluster(embeddings, labels)
print(cluster_stats)

# %%
