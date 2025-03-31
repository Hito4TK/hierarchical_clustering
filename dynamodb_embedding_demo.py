#%%
import boto3
import json
import time

from decimal import Decimal

# Float から Decimal への変換
def convert_float_to_decimal(embedding_list):
    return [Decimal(str(item)) for item in embedding_list]

# Decimal を float に変換
def convert_decimal_to_float(embedding_list):
    return [float(item) for item in embedding_list]


# ------------------------------
# AWS クライアントの作成
# ------------------------------
# DynamoDB クライアントとリソース
dynamodb_client = boto3.client("dynamodb", region_name="ap-northeast-1")
dynamodb_resource = boto3.resource("dynamodb", region_name="ap-northeast-1")

# Bedrock クライアント
bedrock_client = boto3.client("bedrock-runtime", region_name="ap-northeast-1")

# ------------------------------
# DynamoDB テーブルの定義
# ------------------------------
table_name = "UserResponses"

# テーブルが存在するか確認
def check_table_exists():
    try:
        dynamodb_client.describe_table(TableName=table_name)
        return True
    except dynamodb_client.exceptions.ResourceNotFoundException:
        return False

# DynamoDB テーブルを作成
def create_dynamodb_table():
    if not check_table_exists():
        print(f"テーブル {table_name} を作成中...")
        
        response = dynamodb_client.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "user_id", "KeyType": "HASH"},  # パーティションキー
            ],
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},  # 文字列型
            ],
            BillingMode="PAY_PER_REQUEST",  # オンデマンドモード
        )
        
        # テーブル作成完了を待機
        waiter = dynamodb_client.get_waiter("table_exists")
        waiter.wait(TableName=table_name)
        print(f"テーブル {table_name} が作成されました。")
    else:
        print(f"テーブル {table_name} はすでに存在します。")

# ------------------------------
# 埋め込み生成関数
# ------------------------------
# テキストから埋め込みを生成
def get_embedding(text):
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json",
    )
    
    # レスポンスの解析
    response_body = json.loads(response["body"].read().decode("utf-8"))
    embedding = response_body.get("embedding", [])
    
    if not embedding:
        raise ValueError("埋め込みの取得に失敗しました。")
    
    return embedding

# ------------------------------
# DynamoDB に埋め込みを保存
# ------------------------------
# DynamoDB に埋め込みを保存
def save_embedding_to_dynamodb(user_id, text):
    table = dynamodb_resource.Table(table_name)
    
    # 埋め込みを生成して Decimal に変換
    embedding = convert_float_to_decimal(get_embedding(text))
    
    # DynamoDB にデータを格納
    response = table.put_item(
        Item={
            "user_id": user_id,
            "response_text": text,
            "embedding": embedding,
        }
    )
    print(f"{user_id} のデータを DynamoDB に格納しました。")

# ------------------------------
# バッチ書き込み (BatchWriteItem)
# ------------------------------
def batch_write_embeddings(items):
    table = dynamodb_resource.Table(table_name)
    with table.batch_writer() as batch:
        for item in items:
            # Float を Decimal に変換
            item["embedding"] = convert_float_to_decimal(item["embedding"])
            batch.put_item(Item=item)
    print(f"{len(items)} 件のデータをバッチで格納しました。")

# ------------------------------
# クエリとスキャン
# ------------------------------
# Query で特定ユーザーのデータを取得
def query_user_data(user_id):
    table = dynamodb_resource.Table(table_name)
    response = table.get_item(Key={"user_id": user_id})

    if "Item" in response:
        item = response["Item"]
        # Decimal → float に変換
        if "embedding" in item:
            item["embedding"] = convert_decimal_to_float(item["embedding"])

        print(f"{user_id} のデータ: {item}")
    else:
        print(f"{user_id} のデータは見つかりませんでした。")

# Scan ですべてのデータを取得 (ページネーション対応)
def scan_all_data(limit=10):
    table = dynamodb_resource.Table(table_name)
    last_evaluated_key = None
    total_items = []
    page_number = 1

    while True:
        scan_kwargs = {
            "Limit": limit,
        }
        if last_evaluated_key:
            scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])

        # Decimal → float に変換
        for item in items:
            if "embedding" in item:
                item["embedding"] = convert_decimal_to_float(item["embedding"])

        total_items.extend(items)
        print(f"ページ {page_number}: {len(items)} 件のデータを取得しました。")
        for item in items:
            print(item)

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break
        page_number += 1

    print(f"合計 {len(total_items)} 件のデータが取得されました。")

# キーワード検索 (FilterExpression で部分一致)
def search_by_keyword(keyword, limit=10):
    table = dynamodb_resource.Table(table_name)
    last_evaluated_key = None
    total_items = []
    page_number = 1

    while True:
        scan_kwargs = {
            "FilterExpression": "contains(response_text, :keyword)",
            "ExpressionAttributeValues": {":keyword": keyword},
            "Limit": limit,
        }
        if last_evaluated_key:
            scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])

        # Decimal → float に変換
        for item in items:
            if "embedding" in item:
                item["embedding"] = convert_decimal_to_float(item["embedding"])

        total_items.extend(items)
        print(f"ページ {page_number}: キーワード '{keyword}' で {len(items)} 件のデータを取得しました。")
        for item in items:
            print(item)

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break
        page_number += 1

    print(f"キーワード '{keyword}' で合計 {len(total_items)} 件のデータが見つかりました。")

# ------------------------------
# サンプルデータの生成
# ------------------------------
def generate_sample_data():
    data_list = []
    base_texts = [
        "I enjoy playing classical guitar.",
        "Reading historical novels is my hobby.",
        "I love eating sushi and ramen.",
        "Photography and traveling are my passions.",
        "I enjoy learning about AI and machine learning.",
    ]
    
    for i in range(1, 26):  # 25 件のデータ生成
        text = base_texts[i % len(base_texts)] + f" Data sample {i}"
        
        # 埋め込みを Decimal に変換して格納
        data_list.append({
            "user_id": f"user_{str(i).zfill(3)}",
            "response_text": text,
            "embedding": convert_float_to_decimal(get_embedding(text)),
        })
        print(f"データ {i} 件目の生成完了: {text}")

    return data_list

# ------------------------------
# メイン関数
# ------------------------------
def main():
    # DynamoDB テーブルの作成
    create_dynamodb_table()

    # 25 件のサンプルデータ生成
    print("25 件のデータを生成中...")
    sample_data = generate_sample_data()

    # バッチ書き込みで 25 件のデータ格納
    batch_write_embeddings(sample_data)

    # クエリ・スキャンの実行
    query_user_data("user_003")  # ユーザー検索
    scan_all_data(limit=10)  # ページネーション対応スキャン
    search_by_keyword("classical", limit=10)  # キーワード検索

# ------------------------------
# スクリプトの実行
# ------------------------------
if __name__ == "__main__":
    main()
# %%
