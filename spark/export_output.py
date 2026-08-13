import json
import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# 1. Khởi tạo Spark Session
spark = SparkSession.builder \
    .appName("ExportBatchOutputs") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Tạo thư mục output nếu chưa có
os.makedirs("output", exist_ok=True)

# 2. Load mô hình ALS
model_path = "model/als_model"
try:
    als_model = ALSModel.load(model_path)
    print("Đã tải mô hình ALS thành công.")

    # 3. Xuất Top 5 gợi ý cho TẤT CẢ User (Batch Recommendation)
    print("Đang tạo gợi ý cho toàn bộ User...")
    all_user_recs = als_model.recommendForAllUsers(5)
    
    # Lưu kết quả ra file Parquet / CSV trong thư mục output/
    output_batch_path = "output/batch_recommendations"
    all_user_recs.write.mode("overwrite").parquet(output_batch_path)
    print(f"Đã xuất dữ liệu gợi ý hàng loạt ra: {output_batch_path}")

    # 4. Xuất file thông số mô hình (metrics.json)
    metrics = {
        "model_type": "ALS",
        "rank": als_model.rank,
        "status": "Success",
        "output_generated": True
    }
    with open("output/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print("Đã tạo file output/metrics.json")

except Exception as e:
    print("Lỗi trong quá trình xuất output:", e)

spark.stop()