#%%
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from math import ceil
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
import pickle
import json
import boto3
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import requests  # ホットペッパーAPI用

# -------------------------------
# クラスタリング関連の関数
# -------------------------------
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
    
    ※大きいクラスタの分割を先に行い、その後に小さいクラスタの統合を行います。
    """
    labels_adjusted = labels.copy()
    
    for iteration in range(max_iter):
        changed = False
        cluster_counts = Counter(labels_adjusted)

        # --- 大きいクラスタの分割 ---
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

        # --- 小さいクラスタの統合 ---
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
        if (end - start) < min_chunk_size and chunks:
            chunks[-1] = np.concatenate([chunks[-1], embeddings[start:end]], axis=0)
            break
        else:
            chunks.append(embeddings[start:end])
        start = end
    return chunks

# -------------------------------
# テストデータ生成＆埋め込み取得
# -------------------------------
def make_testdata():
    """
    2000人分の「好きな食べ物・好きな飲み物・居住地・勤務地」データを作成し、
    各人のレスポンスに対して Amazon Titan 埋め込みを取得します。
    
    ・居住地・勤務地は東京都内の全市区町村（諸島部を除く）を含む。
    ・好きな飲み物はバリエーション豊富なリストを用います。
    
    生成したembeddingsとresponsesはローカルファイル（"testdata.pkl"）に保存し、
    次回以降はそのファイルを読み込むようにします。
    """
    filename = "testdata.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        data["embeddings"] = np.array(data["embeddings"])
        return data["embeddings"], data["responses"]

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

    drinks = [
        "コーヒー", "紅茶", "緑茶", "ウーロン茶", "オレンジジュース", "炭酸水", "牛乳", "豆乳",
        "スポーツドリンク", "ビール", "ワイン", "日本酒", "焼酎", "ハイボール", "カクテル",
        "ミルクシェイク", "フルーツスムージー", "抹茶ラテ", "カフェラテ", "エスプレッソ", "アイスティー", "レモネード", "ホットチョコレート"
    ]

    residences = [
        # 23特別区
        "千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区", "品川区", "目黒区",
        "大田区", "世田谷区", "渋谷区", "中野区", "杉並区", "豊島区", "北区", "荒川区", "板橋区", "練馬区",
        "足立区", "葛飾区", "江戸川区",
        # 26市（代表例）
        "八王子市", "立川市", "武蔵野市", "三鷹市", "小金井市", "小平市", "府中市", "昭島市", "調布市",
        "町田市", "日野市", "東村山市", "国分寺市", "国立市", "福生市", "多摩市", "青梅市", "羽村市", "西東京市", "狛江市",
        # 本州の町村（諸島部除く）
        "奥多摩町", "檜原村"
    ]
    workplaces = residences.copy()

    num_people = 100  # テスト用 100サンプル
    responses = []
    for _ in range(num_people):
        fav_food = " ".join(random.sample(food_items, 3))
        fav_drink = " ".join(random.sample(drinks, 3))
        residence = random.choice(residences)
        workplace = random.choice(workplaces)
        response = f"{fav_food} {fav_drink} {residence} {workplace}"
        responses.append(response)

    client = boto3.client("bedrock-runtime", region_name="us-east-2")
    model_id = "amazon.titan-embed-text-v2:0"

    embeddings = []
    for text in responses:
        native_request = {"inputText": text}
        response_boto = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
        result = json.loads(response_boto['body'].read())
        embedding = result['embedding']
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    with open("testdata.pkl", "wb") as f:
        pickle.dump({"embeddings": embeddings, "responses": responses}, f)

    return embeddings, responses

# -------------------------------
# Claude 3.5 Haiku によるクラスタ名称生成関連の関数
# -------------------------------
def query_claude(prompt, model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0"):
    """
    Claude 3.5 Haiku API を呼び出し、プロンプトに基づく応答から「text」部分のみを抽出して返す関数です。
    """
    client = boto3.client("bedrock-runtime", region_name="us-east-2")
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]
    inference_config = {"maxTokens": 200, "temperature": 0}
    
    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_config
        )
        # response['body'] がバイト列の場合は読み込み、または response がすでに dict の場合も想定
        if isinstance(response, dict) and "body" in response:
            result = json.loads(response['body'].read())
        else:
            result = response

        # "output" -> "message" -> "content" 内の"text"を抽出
        output = result.get("output", {})
        message = output.get("message", {})
        content_list = message.get("content", [])
        text = " ".join([item.get("text", "") for item in content_list])
        return text.strip()
    except Exception as e:
        print("エラーが発生しました:", e)
        return None

def sample_cluster_name(clusters):
    """
    clusters 辞書（クラスタID -> [(サンプルID, response), ...]）から、
    各クラスタ内の回答3件をランダムに抽出し、
    Claude 3.5 Haiku API に問い合わせてクラスタ名称の提案を取得する関数です。
    """
    cluster_names = {}
    for cluster_id, items in clusters.items():
        sample_items = random.sample(items, min(3, len(items)))
        sample_texts = [response for _, response in sample_items]
        prompt = (
            f"以下は、あるクラスタに属する3人の回答です。\n"
            f"1. {sample_texts[0]}\n"
            f"2. {sample_texts[1]}\n"
            f"3. {sample_texts[2]}\n\n"
            f"これらの回答内容から、このクラスタを表す適切な名称を日本語で提案してください。\n"
            f"考えた名称のみを回答してください。他の情報は不要です。"
        )
        # print(f"クラスタ {cluster_id} のプロンプト:\n{prompt}\n")
        response_text = query_claude(prompt)
        if response_text:
            cluster_names[cluster_id] = response_text
            # print(f"クラスタ {cluster_id} の名称提案: {response_text}\n")
        else:
            cluster_names[cluster_id] = "エラー"
    return cluster_names

# -------------------------------
# ホットペッパーAPI／ツール関連の関数
# -------------------------------
# ホットペッパーAPIの設定
HOTPEPPER_API_KEY = os.getenv('API_KEY')  # 環境変数からAPIキーを取得してください
HOTPEPPER_API_URL = 'http://webservice.recruit.co.jp/hotpepper/gourmet/v1/'

def fetch_restaurant_info(keyword, location):
    """
    ホットペッパーAPIを呼び出して、指定されたキーワードと場所に基づくレストラン情報を取得する。
    """
    params = {
        'key': HOTPEPPER_API_KEY,
        'keyword': keyword,
        'address': location,
        'format': 'json'
    }
    response = requests.get(HOTPEPPER_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and 'shop' in data['results']:
            return data['results']['shop']
    return []

def recommend_restaurant_for_cluster(cluster_items):
    """
    クラスタ内の全員の「好きな食べ物・飲み物」および居住地・勤務地の情報を考慮し、
    おすすめの飲食店をホットペッパーAPIで取得する関数です。
    
    各回答は "食べ物3項目 飲み物3項目 居住地 勤務地" の形式と仮定しています。
    """
    # 3名をランダムサンプリングして食べ物情報を抽出
    sample_items = random.sample(cluster_items, min(3, len(cluster_items)))
    food_candidates = []
    for _, response in sample_items:
        tokens = response.split()
        if len(tokens) >= 8:
            # tokens[0:3] が「好きな食べ物」
            food_candidates.extend(tokens[0:3])
    food_counts = Counter(food_candidates)
    if food_counts:
        food_keyword = food_counts.most_common(1)[0][0]
    else:
        food_keyword = "料理"
    
    # クラスタ内の全回答から居住地・勤務地を抽出して、最頻出のものを採用
    location_candidates = []
    for _, response in cluster_items:
        tokens = response.split()
        if len(tokens) >= 8:
            location_candidates.append(tokens[6])  # 居住地
            location_candidates.append(tokens[7])  # 勤務地
    location_counts = Counter(location_candidates)
    if location_counts:
        location_keyword = location_counts.most_common(1)[0][0]
    else:
        location_keyword = "東京"
    
    # クエリ例: 「{location_keyword}でおすすめの{food_keyword}屋を教えてください。」
    query = f"{location_keyword}でおすすめの{food_keyword}屋を教えてください。"
    # 実際には、fetch_restaurant_info を呼び出してレストラン情報を取得
    restaurants = fetch_restaurant_info(food_keyword, location_keyword)
    if restaurants:
        restaurant_names = [shop.get('name', '不明') for shop in restaurants if shop.get('name')]
        recommendation = "おすすめのレストラン: " + ", ".join(restaurant_names)
    else:
        recommendation = "レストラン情報が見つかりませんでした。"
    
    return recommendation

def sample_restaurant_recommendation(clusters):
    """
    各クラスタについて、recommend_restaurant_for_cluster を呼び出し、
    飲食店の提案を取得する関数です。
    """
    restaurant_recommendations = {}
    for cluster_id, items in clusters.items():
        recommendation = recommend_restaurant_for_cluster(items)
        restaurant_recommendations[cluster_id] = recommendation
    return restaurant_recommendations

# -------------------------------
# メイン処理
# -------------------------------
if __name__ == "__main__":
    # テストデータの作成：embeddings と responses の両方を取得
    embeddings, responses = make_testdata()
    total_samples = len(embeddings)
    print(f"全サンプル数: {total_samples}")

    # クラスタ内のサンプル数の下限・上限
    min_cluster_size = 3
    max_cluster_size = 7

    # サンプル数が多い場合はチャンクに分割（今回は100サンプルなのでそのまま）
    if total_samples > 200:
        chunks = split_embeddings(embeddings, min_chunk_size=50, max_chunk_size=200)
    else:
        chunks = [embeddings]

    global_labels_list = []
    global_label_offset = 0

    # 各チャンク毎にクラスタリングを実施
    for chunk in chunks:
        chunk_size = len(chunk)
        init_n_clusters = max(1, int(round(chunk_size * 0.1)))
        labels = hierarchical_clustering_with_constraints(
            chunk, min_cluster_size, max_cluster_size, n_clusters=init_n_clusters
        )
        labels_adjusted = labels + global_label_offset
        global_labels_list.append(labels_adjusted)
        global_label_offset += (labels.max() + 1)

    adjusted_labels = np.concatenate(global_labels_list)

    # 各クラスタのサンプル数を表示
    counts = Counter(adjusted_labels)
    print("最終的なクラスタサイズ:")
    for lab, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  クラスタ {lab}: {cnt} サンプル")

    # クラスタ毎に、各人のIDと回答をまとめる
    clusters = {}
    for i, label in enumerate(adjusted_labels):
        clusters.setdefault(label, []).append((i, responses[i]))

    # ここで、各クラスタから3名をサンプリングして名称を生成
    cluster_name_suggestions = sample_cluster_name(clusters)
    # 各クラスタに対して、飲食店の提案も取得する
    restaurant_recommendations = sample_restaurant_recommendation(clusters)

    # 各クラスタの内容と、Claudeの名称提案＋飲食店提案を表示
    for label, items in clusters.items():
        # cluster_name_suggestions から名称を取得（無ければ "名称なし"）
        cluster_name = cluster_name_suggestions.get(label, "名称なし")
        restaurant_rec = restaurant_recommendations.get(label, "飲食店情報なし")
        print(f"\nクラスタ {label} ({len(items)} 件) - 提案名称: {cluster_name} / 飲食店提案: {restaurant_rec}")
        for person_id, response in items:
            print(f"  ID {person_id}: {response}")

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
