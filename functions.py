#%%
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from collections import defaultdict
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
# ホットペッパーAPI／ツール関連の関数
# -------------------------------
# ホットペッパーAPIの設定
HOTPEPPER_API_KEY = os.getenv('API_KEY')  # 環境変数からAPIキーを取得
HOTPEPPER_API_URL = 'http://webservice.recruit.co.jp/hotpepper/gourmet/v1/'

def fetch_restaurant_info(keyword, location):
    """
    ホットペッパーAPIを呼び出して、指定されたキーワードと場所に基づくレストラン情報を取得する関数です。
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

# -------------------------------
# （以下、既存の関数群）
# -------------------------------

def hierarchical_clustering_with_constraints(embeddings, min_cluster_size, max_cluster_size, n_clusters):
    """
    AgglomerativeClustering を用いてコサイン類似度に基づいたクラスタリングを行い、
    クラスタ内の人数が [min_cluster_size, max_cluster_size] の範囲に収まるように調整する関数です。
    """
    cosine_distances = 1 - cosine_similarity(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    initial_labels = clustering.fit_predict(cosine_distances)
    adjusted_labels = adjust_cluster_sizes_iterative(
        labels=initial_labels,
        embeddings=embeddings,
        min_size=min_cluster_size,
        max_size=max_cluster_size,
        max_iter=10
    )
    return adjusted_labels

def adjust_cluster_sizes_iterative(labels, embeddings, min_size, max_size, max_iter=1000):
    labels_adjusted = labels.copy()
    for iteration in range(max_iter):
        changed = False
        cluster_counts = Counter(labels_adjusted)
        # 大きいクラスタの分割
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
        # 小さいクラスタの統合
        '''
        cluster_counts = Counter(labels_adjusted)
        for cluster, size in list(cluster_counts.items()):
            if size < min_size:
                target_cluster = find_closest_cluster(cluster, labels_adjusted, embeddings)
                if target_cluster is not None and target_cluster != cluster:
                    labels_adjusted[labels_adjusted == cluster] = target_cluster
                    changed = True
        '''
        if not changed:
            break
    return labels_adjusted

def find_closest_cluster(target_cluster, labels, embeddings):
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
    ランダムな順序でembeddingをチャンクに分割し、元のインデックスも保持する関数。
    
    Args:
        embeddings (numpy.ndarray): ベクトル化されたデータ
        min_chunk_size (int): 最小チャンクサイズ
        max_chunk_size (int): 最大チャンクサイズ

    Returns:
        list of tuples: [(chunk_indices, chunk_embeddings), ...]
    """
    n_samples = len(embeddings)
    indices = np.arange(n_samples)  # 元のインデックス
    np.random.shuffle(indices)  # ランダムな順番にシャッフル

    chunks = []
    start = 0
    while start < n_samples:
        end = min(start + max_chunk_size, n_samples)
        if (end - start) < min_chunk_size and chunks:
            # 最後のチャンクが小さすぎる場合は、前のチャンクに統合
            last_indices, last_embeddings = chunks[-1]
            new_indices = np.concatenate([last_indices, indices[start:end]])
            new_embeddings = np.concatenate([last_embeddings, embeddings[indices[start:end]]], axis=0)
            chunks[-1] = (new_indices, new_embeddings)
            break
        else:
            chunk_indices = indices[start:end]
            chunk_embeddings = embeddings[chunk_indices]
            chunks.append((chunk_indices, chunk_embeddings))
        start = end
    
    return chunks


def make_testdata():
    filename = "testdata_food.pkl"

    # すでにデータがある場合は読み込み
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        data["embeddings"] = np.array(data["embeddings"])
        return data["embeddings"], data["responses"]

    # 乱数シードの固定（再現性確保）
    np.random.seed(42)
    random.seed(42)

    # 食べ物、飲み物、駅の候補リスト
    food_items = [
        "寿司", "ラーメン", "うどん", "そば", "焼肉"
    ]
    drinks = [ "コーヒー", "紅茶", "緑茶", "ウーロン茶", "オレンジジュース", "炭酸水", "牛乳", "豆乳" ]
    residences = [ "門前仲町駅", "晴海駅", "飯田橋駅", "早稲田駅", "東京駅" ]
    workplaces = residences.copy()

    # 質問リスト
    questions = [
        "好きな食べ物は何ですか？",
        "好きな飲み物は何ですか？",
        "住んでいる場所はどこですか？",
        "働いている場所はどこですか？"
    ]

    # データ生成数
    num_people = 10
    responses = []

    # 各人の回答を生成
    for i in range(num_people):
        fav_food = " ".join(random.sample(food_items, 2))
        fav_drink = " ".join(random.sample(drinks, 2))
        residence = random.choice(residences)
        workplace = random.choice(workplaces)

        # 質問ごとの回答を辞書形式で格納
        person_responses = [
            {"person_id": i, "question": questions[0], "response": fav_food},
            {"person_id": i, "question": questions[1], "response": fav_drink},
            {"person_id": i, "question": questions[2], "response": residence},
            {"person_id": i, "question": questions[3], "response": workplace}
        ]

        # 全回答をresponsesに追加
        responses.extend(person_responses)

    # 必要な質問だけ選択する
    selected_questions = [
        "好きな食べ物は何ですか？",
        "住んでいる場所はどこですか？"
    ]

    # person_id ごとに必要な回答だけをまとめる
    person_responses_dict = defaultdict(list)
    for item in responses:
        if item["question"] in selected_questions:
            combined_text = f"{item['question']} {item['response']}"
            person_responses_dict[item["person_id"]].append(combined_text)

    # Bedrockクライアントの作成
    client = boto3.client("bedrock-runtime", region_name="ap-northeast-1")
    model_id = "amazon.titan-embed-text-v2:0"

    # 同じ人の回答をまとめて埋め込み対象に
    embeddings = []
    grouped_responses = []
    for person_id, texts in person_responses_dict.items():
        combined_text = " ".join(texts)
        
        # Bedrockへのリクエスト
        native_request = {"inputText": combined_text}
        response_boto = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
        result = json.loads(response_boto["body"].read())

        # 埋め込みデータ取得
        embedding = result["embedding"]
        embeddings.append(embedding)

        # まとめた回答（combined_text）のみを保存
        grouped_responses.append(combined_text)

    # 埋め込みデータをNumPy配列に変換
    embeddings = np.array(embeddings)

    # データを pickle に保存
    with open("testdata_food.pkl", "wb") as f:
        pickle.dump({"embeddings": embeddings, "responses": grouped_responses}, f)

    return embeddings, grouped_responses

def query_claude(prompt, model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", max_retries=5):
    client = boto3.client("bedrock-runtime", region_name="ap-northeast-1")
    
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]
    
    inference_config = {"maxTokens": 200, "temperature": 0}
    
    retries = 0
    while retries < max_retries:
        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig=inference_config
            )
            
            # レスポンスが辞書形式かどうか確認
            if isinstance(response, dict) and "body" in response:
                result = json.loads(response['body'].read())
            else:
                result = response
            
            output = result.get("output", {})
            message = output.get("message", {})
            content_list = message.get("content", [])
            text = " ".join([item.get("text", "") for item in content_list])
            return text.strip()
        
        except client.exceptions.ThrottlingException:
            # Throttling時の待機時間を指数関数的に増やす
            wait_time = 2 ** retries + random.uniform(0, 1)
            print(f"リクエスト制限超過。{wait_time:.2f}秒待機して再試行します...")
            time.sleep(wait_time)
            retries += 1
        
        except Exception as e:
            print("エラーが発生しました:", e)
            return None

    print("最大リトライ回数に達しました。リクエスト失敗。")
    return None

def sample_cluster_name(clusters):
    cluster_names = {}
    
    for cluster_id, items in clusters.items():
        # クラスタ内の回答からランダムに3つサンプル
        sample_items = random.sample(items, min(3, len(items)))
        sample_texts = [response for _, response in sample_items]
        
        # プロンプト生成
        prompt = (
            f"以下は、あるクラスタに属する3人の回答です。\n"
            f"1. {sample_texts[0]}\n"
            f"2. {sample_texts[1]}\n"
            f"3. {sample_texts[2]}\n\n"
            f"これらの回答内容から、このクラスタを表す適切な名称を日本語で提案してください。\n"
            f"考えた名称のみを回答してください。他の情報は不要です。"
        )
        
        # Claude へのリクエスト
        response_text = query_claude(prompt)
        
        if response_text:
            cluster_names[cluster_id] = response_text
        else:
            cluster_names[cluster_id] = "エラー"

        # APIリクエストの頻度を抑制（1〜2秒の間隔）
        time.sleep(random.uniform(1, 2))
    
    return cluster_names

# -------------------------------
# Converse APIによるfunction callingを利用したツール呼び出し
# -------------------------------
tools = [
    {
        "name": "fetch_restaurant_info",
        "description": "指定されたキーワードと場所に基づいてレストラン情報を取得します。",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "例：寿司、ラーメン"},
                "location": {"type": "string", "description": "例：東京、新宿"}
            },
            "required": ["keyword", "location"]
        }
    }
]

import time  # 待機用

def call_restaurant_tool(query, model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"):
    client = boto3.client('bedrock-runtime', region_name='ap-northeast-1')
    initial_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": query}]}
        ],
        "tools": tools
    }
    
    max_retries = 5
    delay = 1  # 初回待機秒数
    for attempt in range(max_retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(initial_payload),
                contentType="application/json"
            )
            break  # 成功したらループを抜ける
        except client.exceptions.ThrottlingException as e:
            print(f"ThrottlingException 発生。{delay}秒待機中... (試行 {attempt+1}/{max_retries})")
            time.sleep(delay)
            delay *= 2  # 指数バックオフ
    else:
        return "APIリクエストがタイムアウトしました。"

    response_body = json.loads(response['body'].read())
    if response_body.get('content') and response_body['content'][0].get('type') == 'tool_use':
        tool_request = response_body['content'][0]
        tool_use_id = tool_request['id']
        tool_name = tool_request['name']
        tool_input = tool_request['input']
        if tool_name == 'fetch_restaurant_info':
            restaurants = fetch_restaurant_info(tool_input.get('keyword'), tool_input.get('location'))
            if restaurants:
                restaurant_names = [shop.get('name', '不明') for shop in restaurants if shop.get('name')]
                tool_result_text = "おすすめのレストラン: " + ", ".join(restaurant_names)
            else:
                tool_result_text = "レストラン情報が見つかりませんでした。"
            followup_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "temperature": 0.5,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": query}]},
                    {"role": "assistant", "content": [tool_request]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": tool_result_text}]}
                ],
                "tools": tools
            }
            # 同様にフォローアップリクエストでもリトライ処理を実施
            delay = 1
            for attempt in range(max_retries):
                try:
                    followup_response = client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(followup_payload),
                        contentType="application/json"
                    )
                    break
                except client.exceptions.ThrottlingException as e:
                    print(f"（フォローアップ）ThrottlingException 発生。{delay}秒待機中... (試行 {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
            else:
                return "フォローアップAPIリクエストがタイムアウトしました。"
            followup_response_body = json.loads(followup_response['body'].read())
            if 'completion' in followup_response_body:
                final_answer = followup_response_body['completion']
            elif 'content' in followup_response_body:
                final_answer = ""
                for block in followup_response_body['content']:
                    if block.get('type') == 'text' and 'text' in block:
                        final_answer += block['text']
            else:
                final_answer = ""
            return final_answer
        else:
            return "要求されたツールが認識できませんでした: " + tool_name
    else:
        if 'completion' in response_body:
            return response_body['completion']
        elif 'content' in response_body:
            final_answer = ""
            for block in response_body['content']:
                if block.get('type') == 'text' and 'text' in block:
                    final_answer += block['text']
            return final_answer
        else:
            return "最終回答が見つかりませんでした。"

def recommend_restaurant_for_cluster(cluster_items):
    # 全クラスタのデータを対象にする
    food_candidates = []
    location_candidates = []

    for _, response in cluster_items:
        lines = response.split(" ")
        for i in range(len(lines) - 1):
            # 食べ物の質問に対する回答
            if "好きな食べ物は何ですか？" in lines[i]:
                food_answers = lines[i + 1].split()
                food_candidates.extend(food_answers)
            
            # 住んでいる場所の質問に対する回答
            elif "住んでいる場所はどこですか？" in lines[i]:
                location_answers = lines[i + 1].split()
                location_candidates.extend(location_answers)

    # 食べ物候補の頻度カウント（すべての回答を考慮）
    food_counts = Counter(food_candidates)
    food_keyword = food_counts.most_common(1)[0][0] if food_counts else "料理"

    # 場所候補の頻度カウント（すべての回答を考慮）
    location_counts = Counter(location_candidates)
    location_keyword = location_counts.most_common(1)[0][0] if location_counts else "東京"

    # APIリクエストの前にランダム待機時間を追加（0.5〜2.0秒）
    wait_time = random.uniform(0.5, 2.0)
    print(f"APIリクエストの前に {wait_time:.2f} 秒待機...")
    time.sleep(wait_time)

    query = f"{location_keyword}でおすすめの{food_keyword}屋を教えてください。"
    print(query)
    recommendation = call_restaurant_tool(query)
    return recommendation

def sample_restaurant_recommendation(clusters):
    restaurant_recommendations = {}
    for cluster_id, items in clusters.items():
        recommendation = recommend_restaurant_for_cluster(items)
        restaurant_recommendations[cluster_id] = recommendation
    return restaurant_recommendations


# -------------------------------
# クラスタごとのサンプル数の集計と、類似度統計量を計算する関数
# -------------------------------
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


# %%