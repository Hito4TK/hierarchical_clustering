#%%
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from math import ceil
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def hierarchical_clustering_with_constraints(embeddings, min_cluster_size, max_cluster_size, n_clusters):
    """
    AgglomerativeClustering を用いてコサイン類似度に基づいたクラスタリングを行い、
    クラスタ内の人数が [min_cluster_size, max_cluster_size] の範囲に収まるように調整する関数です。
    """
    # 事前にコサイン類似度を計算し、距離（1 - 類似度）に変換
    cosine_distances = 1 - cosine_similarity(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    initial_labels = clustering.fit_predict(cosine_distances)

    # 初期ラベルに対してサイズ制約を反復的に調整
    adjusted_labels = adjust_cluster_sizes_iterative(labels=initial_labels,
                                                     embeddings=embeddings,
                                                     min_size=min_cluster_size,
                                                     max_size=max_cluster_size,
                                                     max_iter=10)
    return adjusted_labels

def adjust_cluster_sizes_iterative(labels, embeddings, min_size, max_size, max_iter=1000):
    """
    クラスタサイズの制約（最小・最大）を反復的に調整します。
    
    ※今回の実装では、**大きいクラスタの分割を先に行い、その後に小さいクラスタの統合を行う**処理順序です。
    
    ・サイズが max_size を超えるクラスタは、まず分割します。  
    ・その後、サイズが min_size 未満のクラスタは、最も近い他クラスタと統合します。
    """
    labels_adjusted = labels.copy()
    
    for iteration in range(max_iter):
        changed = False
        cluster_counts = Counter(labels_adjusted)

        # --- まず大きいクラスタの分割を実施 ---
        for cluster, size in list(cluster_counts.items()):
            if size > max_size:
                indices = np.where(labels_adjusted == cluster)[0]
                sub_embeddings = embeddings[indices]
                cosine_distances_sub = 1 - cosine_similarity(sub_embeddings)
                n_subclusters = ceil(size / max_size)
                if n_subclusters < 2:
                    continue
                sub_clustering = AgglomerativeClustering(
                    n_clusters=n_subclusters,
                    metric="precomputed",
                    linkage="average"
                )
                sub_labels = sub_clustering.fit_predict(cosine_distances_sub)
                new_label_offset = max(labels_adjusted) + 1
                for sub_lab in np.unique(sub_labels):
                    sub_indices = indices[sub_labels == sub_lab]
                    labels_adjusted[sub_indices] = new_label_offset
                    new_label_offset += 1
                changed = True

        # --- 次に小さいクラスタの統合を実施 ---
        cluster_counts = Counter(labels_adjusted)
        for cluster, size in list(cluster_counts.items()):
            if size < min_size:
                target_cluster = find_closest_cluster(cluster, labels_adjusted, embeddings)
                if target_cluster is not None and target_cluster != cluster:
                    labels_adjusted[labels_adjusted == cluster] = target_cluster
                    changed = True

        if not changed:
            break

    return labels_adjusted

def find_closest_cluster(target_cluster, labels, embeddings):
    """
    指定クラスタ(target_cluster)とその他クラスタとの平均埋め込み間のコサイン類似度を計算し、
    最も類似度が高いクラスタを返します。
    """
    indices_target = np.where(labels == target_cluster)[0]
    if len(indices_target) == 0:
        return None
    vec_target = np.mean(embeddings[indices_target], axis=0).reshape(1, -1)

    best_cluster = None
    best_sim = -1
    for other_cluster in set(labels):
        if other_cluster == target_cluster:
            continue
        indices_other = np.where(labels == other_cluster)[0]
        vec_other = np.mean(embeddings[indices_other], axis=0).reshape(1, -1)
        sim = cosine_similarity(vec_target, vec_other)[0, 0]
        if sim > best_sim:
            best_sim = sim
            best_cluster = other_cluster

    return best_cluster

def split_embeddings(embeddings, min_chunk_size=50, max_chunk_size=200):
    """
    embeddings をチャンクに分割します。
    各チャンクはできるだけ max_chunk_size に近く、かつ
    最後のチャンクが min_chunk_size 未満の場合は前のチャンクと結合します。
    """
    n_samples = len(embeddings)
    if n_samples <= max_chunk_size:
        return [embeddings]

    chunks = []
    start = 0
    while start < n_samples:
        end = min(start + max_chunk_size, n_samples)
        # 最後のチャンクが min_chunk_size 未満の場合、直前のチャンクと結合
        if (end - start) < min_chunk_size and chunks:
            chunks[-1] = np.concatenate([chunks[-1], embeddings[start:end]], axis=0)
            break
        else:
            chunks.append(embeddings[start:end])
        start = end
    return chunks

def make_testdata():
    """
    2000人分の「好きな食べ物」データを作成します。
    各人はランダムに3種類の食べ物を選び、TF-IDF と次元削減で埋め込みを作成します。
    この関数は (embeddings, responses) のタプルを返します。
    """
    np.random.seed(42)
    random.seed(42)

    food_items = [
        "寿司", "ラーメン", "うどん", "そば", "焼肉", "カレー", "ピザ", "パスタ", "ハンバーガー", "餃子", 
        "唐揚げ", "刺身", "天ぷら", "しゃぶしゃぶ", "すき焼き", "ステーキ", "オムライス", "ドーナツ", "ケーキ", "チョコレート", 
        "アイスクリーム", "フルーツタルト", "パンケーキ", "タピオカ", "ナポリタン", "カルボナーラ", "エビフライ", "カツ丼", "親子丼", "ビーフシチュー", 
        "焼き鳥", "フォー", "バインミー", "キムチ", "チーズ", "ホットドッグ", "サンドイッチ", "クレープ", "抹茶スイーツ", "杏仁豆腐", 
        "パフェ", "たこ焼き", "お好み焼き", "もんじゃ焼き", "牛丼", "天丼", "ミートソース", "グラタン", "ドリア", "シチュー", 
        "コロッケ", "春巻き", "チャーハン", "中華まん", "麻婆豆腐", "青椒肉絲", "北京ダック", "担々麺", "冷麺", "キンパ", 
        "サムギョプサル", "ビビンバ", "トッポギ", "カオマンガイ", "ガパオライス", "トムヤムクン", "ナシゴレン", "サテ", "バターチキンカレー", "タンドリーチキン", 
        "ナン", "チキンティッカ", "ケバブ", "フムス", "ファラフェル", "シュクシュカ", "ボルシチ", "ピロシキ", "ラタトゥイユ", "カプレーゼ", 
        "リゾット", "ペスカトーレ", "ジェノベーゼ", "ガーリックシュリンプ", "ローストビーフ", "ローストチキン", "スペアリブ", "フィッシュアンドチップス", "バーベキュー", "ミートパイ", 
        "ビスケット", "パンナコッタ", "ティラミス", "マカロン", "エクレア", "プリン", "バウムクーヘン", "ショートケーキ", "ミルフィーユ", "モンブラン", 
        "アップルパイ", "スイートポテト", "クッキー", "チュロス", "ゼリー", "たい焼き", "どら焼き", "わらび餅", "みたらし団子", "大福", 
        "羊羹", "かき氷", "クリームソーダ", "レモネード", "ココア", "抹茶ラテ", "カフェラテ", "エスプレッソ", "フルーツジュース", "スムージー",
        "コーンポタージュ", "ミネストローネ", "オニオンスープ", "ガスパチョ", "ブイヤベース", "ハヤシライス", "ロコモコ", "チリコンカン", "ジャンバラヤ", "パエリア", 
        "ソーセージ", "ハム", "ベーコン", "ロースハム", "生ハム", "スモークサーモン", "エビチリ", "フカヒレスープ", "ショウロンポウ", "チャーシュー",
        "焼きそば", "焼きうどん", "カレーうどん", "皿うどん", "タコライス", "ゴーヤチャンプルー", "サーターアンダギー", "沖縄そば", "シークワーサージュース", "マンゴープリン",
        "ヨーグルト", "シリアル", "グラノーラ", "バター", "ジャム", "はちみつ", "あんこ", "きなこ", "黒蜜", "豆乳",
        "ほうじ茶", "玄米茶", "ジャスミン茶", "ウーロン茶", "紅茶", "緑茶", "麦茶", "ハーブティー", "ミルクティー", "チャイ"
    ]

    # 各人はランダムに3種類の食べ物を選択
    num_people = 2000
    responses = [" ".join(random.sample(food_items, 3)) for _ in range(num_people)]

    # TF-IDF によりテキストを数値化し、次元削減で埋め込みを作成
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(responses)

    n_components = min(128, tfidf_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    # 正規化（通常の LLM の embedding と同様）
    #embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings, responses

#%%

if __name__ == "__main__":
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # テストデータの作成：embeddings と回答（選んだ食べ物の文字列）の両方を取得
    embeddings, responses = make_testdata()
    total_samples = len(embeddings)
    print(f"全サンプル数: {total_samples}")

    # クラスタ内のサンプル数の下限・上限
    min_cluster_size = 3
    max_cluster_size = 8

    # サンプル数が多い場合、チャンクサイズを min=50, max=200 に分割
    if total_samples > 200:
        chunks = split_embeddings(embeddings, min_chunk_size=50, max_chunk_size=200)
    else:
        chunks = [embeddings]

    global_labels_list = []
    global_label_offset = 0

    # 各チャンク毎にクラスタリングを実施
    for chunk in chunks:
        chunk_size = len(chunk)
        # 初期クラスタ数をチャンクサイズの約15%（最低1）に設定
        init_n_clusters = max(1, int(round(chunk_size * 0.15)))
        labels = hierarchical_clustering_with_constraints(
            chunk, min_cluster_size, max_cluster_size, n_clusters=init_n_clusters
        )
        # 各チャンクのラベルにグローバルなオフセットを加え、一意にする
        labels_adjusted = labels + global_label_offset
        global_labels_list.append(labels_adjusted)
        global_label_offset += (labels.max() + 1)

    # チャンク毎のラベルを連結
    adjusted_labels = np.concatenate(global_labels_list)

    # 各クラスタのサンプル数を表示
    counts = Counter(adjusted_labels)
    print("最終的なクラスタサイズ:")
    for lab, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  クラスタ {lab}: {cnt} サンプル")

    # クラスタ毎に、各人のID（サンプルインデックス）と回答（選んだ食べ物の文字列）をまとめる
    clusters = {}
    for i, label in enumerate(adjusted_labels):
        clusters.setdefault(label, []).append((i, responses[i]))

    for label, items in clusters.items():
        print(f"\nクラスタ {label} ({len(items)} 件):")
        for person_id, response in items:
            print(f"  ID {person_id}: {response}")

    # クラスタ内でのID重複チェック
    duplicate_found = False
    for label, items in clusters.items():
        ids = [person_id for person_id, _ in items]
        if len(ids) != len(set(ids)):
            print(f"警告: クラスタ {label} 内に重複したIDが存在します!")
            duplicate_found = True
    if not duplicate_found:
        print("\n各クラスタ内でIDの重複はありません。")

    # 全クラスタを通してのID重複チェック
    all_ids = []
    for items in clusters.values():
        for person_id, _ in items:
            all_ids.append(person_id)
    if len(all_ids) != len(set(all_ids)):
        print("警告: 全クラスタを通して重複したIDが存在します!")
    else:
        print("全クラスタを通してIDの重複はありません。")

    # 2次元プロット（PCA による次元削減）
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(3, 2))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=adjusted_labels, cmap="tab10")
    plt.title("クラスタリング結果 (調整後)")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.colorbar(scatter, label="クラスタID")
    plt.tight_layout()
    plt.show()
# %%