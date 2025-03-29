#%%
import numpy as np
import pandas as pd
from collections import Counter
import sys
from functions import fetch_restaurant_info
from functions import hierarchical_clustering_with_constraints
from functions import adjust_cluster_sizes_iterative
from functions import find_closest_cluster
from functions import split_embeddings
from functions import make_testdata
from functions import query_claude
from functions import sample_cluster_name
from functions import call_restaurant_tool
from functions import recommend_restaurant_for_cluster
from functions import sample_restaurant_recommendation
from functions import cluster_cohesion_stats_by_cluster

# -------------------------------
# メイン処理
# -------------------------------
if __name__ == "__main__":
    embeddings, responses = make_testdata()
    total_samples = len(embeddings)

    # 出力をファイルにリダイレクト
    with open("output.txt", "w", encoding="utf-8") as f:
        sys.stdout = f  # 標準出力をファイルに変更

        print(f"全サンプル数: {total_samples}")

        # responsesの出力（クラスタリング前のデータ確認用）
        print("\n取得したresponses:")
        for i, response in enumerate(responses):
            print(f"{i}: {response}")

        min_cluster_size = 3
        max_cluster_size = 7

        # 埋め込みデータをランダムにチャンク分割（元のインデックスを保持）
        if total_samples > 200:
            chunks = split_embeddings(embeddings, min_chunk_size=50, max_chunk_size=200)
        else:
            chunks = [(np.arange(total_samples), embeddings)]  # 全データを1チャンクとして処理

        global_labels_list = []
        global_indices_list = []  # 各チャンクの元のインデックスを保存
        global_label_offset = 0

        for chunk_indices, chunk_embeddings in chunks:
            chunk_size = len(chunk_embeddings)
            init_n_clusters = max(1, int(round(chunk_size * 0.1)))

            # クラスタリング
            labels = hierarchical_clustering_with_constraints(
                chunk_embeddings, min_cluster_size, max_cluster_size, n_clusters=init_n_clusters
            )

            # グローバルなラベル番号に変換
            labels_adjusted = labels + global_label_offset
            global_labels_list.append(labels_adjusted)
            global_indices_list.append(chunk_indices)  # チャンク内の元のインデックスを保存
            global_label_offset += (labels.max() + 1)
            
        # 全チャンクの結果を統合
        adjusted_labels = np.concatenate(global_labels_list)
        original_indices = np.concatenate(global_indices_list)  # 元のインデックスを復元

        # クラスタリング結果を元のデータ順に並び替え
        sorted_indices = np.argsort(original_indices)
        adjusted_labels = adjusted_labels[sorted_indices]

        # クラスタごとのコサイン類似度統計量とサンプル数を計算
        embeddings_np = np.array(embeddings)  # 必要に応じて np.array に変換
        cohesion_stats_by_cluster = cluster_cohesion_stats_by_cluster(embeddings_np, adjusted_labels)

        # クラスタデータを格納するリスト
        cluster_data = []

        # クラスタサイズとコサイン類似度の統計量を表示、保存
        print("最終的なクラスタサイズとコサイン類似度の統計量、データフレームへの格納:")
        for lab, stats in sorted(cohesion_stats_by_cluster.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  クラスタ {lab}: {stats['count']} サンプル")
            print(f"    平均: {stats['mean']:.4f}, 標準偏差: {stats['std']:.4f}, 分散: {stats['var']:.4f}, "
                f"最大値: {stats['max']:.4f}, 最小値: {stats['min']:.4f}, 中央値: {stats['median']:.4f}")

            
            # DataFrame に格納するための辞書を作成
            cluster_data.append({
                'cluster_id': lab,
                'count': stats['count'],
                'mean': stats['mean'],
                'std': stats['std'],
                'var': stats['var'],
                'max': stats['max'],
                'min': stats['min'],
                'median': stats['median']
            })

        # リストから DataFrame を作成
        cluster_df = pd.DataFrame(cluster_data)

        # DataFrame を pkl 形式で保存
        cluster_df.to_pickle('cluster_stats.pkl')
        print(cluster_df.head())

        # 保存完了の確認
        print("\nクラスタ統計情報が 'cluster_stats.pkl' に保存されました。")


        clusters = {}
        for i, label in enumerate(adjusted_labels):
            clusters.setdefault(label, []).append((i, responses[i]))

        # クラスタ内のデータを表示（対応確認用）
        print("\nクラスタ内のデータ:")
        for cluster_id, members in clusters.items():
            print(f"\nクラスタ {cluster_id}:")
            for member_id, response in members:
                print(f"  ID {member_id}: {response}")

        '''
        cluster_name_suggestions = sample_cluster_name(clusters)
        restaurant_recommendations = sample_restaurant_recommendation(clusters)
        for label, items in clusters.items():
            cluster_name = cluster_name_suggestions.get(label, "名称なし")
            restaurant_rec = restaurant_recommendations.get(label, "飲食店情報なし")
            print(f"\nクラスタ {label} ({len(items)} 件) - 提案名称: {cluster_name} / 飲食店提案: {restaurant_rec}")
            for person_id, response in items:
                print(f"  ID {person_id}: {response}")
        '''
    sys.stdout = sys.__stdout__

print("処理が完了しました。")
# %%