from flask import Flask, jsonify, render_template, request
import os
import json
import pandas as pd
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# Xử lý đường dẫn tương đối trỏ chính xác vào dataset/ml-20m/movies.csv
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
movies_csv_path = os.path.join(base_dir, "dataset", "ml-20m", "movies.csv")
model_path = os.path.join(base_dir, "model", "als_model")

# 1. Đọc danh sách phim từ dataset/ml-20m/movies.csv
movies_dict = {}
if os.path.exists(movies_csv_path):
    try:
        movies_df = pd.read_csv(movies_csv_path)
        movies_dict = dict(zip(movies_df['movieId'].astype(int), movies_df['title']))
        print("Đã tải danh sách phim thành công từ:", movies_csv_path)
    except Exception as e:
        print("Không thể đọc file movies.csv:", e)
else:
    print("Không tìm thấy file:", movies_csv_path)

# 2. Khởi tạo Spark Session
spark = SparkSession.builder \
    .appName("FlaskALSRecommendation") \
    .config("spark.driver.memory", "2g") \
    .config("spark.hadoop.io.nativeio.NativeIO$Windows", "false") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 3. Load mô hình ALS
try:
    als_model = ALSModel.load(model_path)
    print("Đã tải thành công mô hình ALS từ:", model_path)
except Exception as e:
    als_model = None
    print("Chưa tìm thấy mô hình ALS:", e)

# 4. Kết nối Redis cache
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    REDIS_AVAILABLE = True
    print("Đã kết nối thành công tới Redis Cache.")
except Exception:
    REDIS_AVAILABLE = False
    print("Cảnh báo: Không thể kết nối Redis Cache. Hệ thống sẽ truy vấn trực tiếp từ Spark.")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend/<int:user_id>", methods=["GET"])
def recommend(user_id):
    cache_key = f"recommendations:{user_id}"
    
    # 1. Kiểm tra cache Redis
    if REDIS_AVAILABLE:
        try:
            cached_data = r.get(cache_key)
            if cached_data:
                parsed_data = json.loads(cached_data)
                if not (len(parsed_data) > 0 and "Movie ID:" in parsed_data[0]):
                    return jsonify({
                        "user": user_id,
                        "source": "redis-cache",
                        "recommendations": parsed_data
                    })
        except Exception as e:
            print("Lỗi khi đọc Redis Cache:", e)

    # 2. Dự đoán từ mô hình Spark ALS
    recommendations = []
    if als_model is not None:
        try:
            single_user_df = spark.createDataFrame([(user_id,)], ["userId"])
            recs = als_model.recommendForUserSubset(single_user_df, 5)
            collected = recs.collect()
            if collected:
                movie_recs = collected[0]["recommendations"]
                
                # Ánh xạ movieId sang Tên phim (Title)
                for item in movie_recs:
                    m_id = int(item['movieId'])
                    rating = round(item['rating'], 2)
                    
                    # Lấy tên phim từ dictionary
                    title = movies_dict.get(m_id, "Tên phim chưa cập nhật")
                    
                    recommendations.append(f"{title} (ID: {m_id}) - Rating dự đoán: {rating}")
        except Exception as e:
            print("Lỗi khi dự đoán từ mô hình ALS:", e)

    # Dự phòng nếu user chưa có trong tập train (Fallback Strategy)
    if not recommendations:
        recommendations = [
            "Shawshank Redemption, The (1994) (ID: 318) - Rating dự đoán: 5.00",
            "Pulp Fiction (1994) (ID: 296) - Rating dự đoán: 5.00",
            "Silence of the Lambs, The (1991) (ID: 593) - Rating dự đoán: 5.00",
            "Forrest Gump (1994) (ID: 356) - Rating dự đoán: 5.00",
            "Matrix, The (1999) (ID: 2571) - Rating dự đoán: 5.00"
        ]

    # 3. Lưu kết quả mới vào Redis cache
    if REDIS_AVAILABLE:
        try:
            r.setex(cache_key, 3600, json.dumps(recommendations, ensure_ascii=False))
        except Exception as e:
            print("Lỗi khi ghi Redis Cache:", e)

    return jsonify({
        "user": user_id,
        "source": "spark-als-model",
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)