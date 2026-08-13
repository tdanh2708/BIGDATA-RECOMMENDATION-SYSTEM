import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os

def main():
    # Khởi tạo Spark Session và cấp phát thêm dung lượng RAM (4GB) cho Driver
    spark = SparkSession.builder \
        .appName("LargeScaleRecommendationALS") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.hadoop.io.nativeio.NativeIO$Windows", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    print("Đang đọc dữ liệu từ thư mục dataset...")
    ratings_df = spark.read.option("header", "true").option("inferSchema", "true").csv("dataset/ml-20m/ratings.csv")

    # Chia dữ liệu thành tập huấn luyện (80%) và kiểm thử (20%)
    (training_data, test_data) = ratings_df.randomSplit([0.8, 0.2], seed=42)

    print("Đang huấn luyện mô hình ALS...")
    als = ALS(
        maxIter=5,
        regParam=0.01,
        rank=10,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(training_data)

    print("Đang đánh giá mô hình trên tập test...")
    predictions = model.transform(test_data)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Square Error (RMSE) = {rmse}")

    # Lưu mô hình đã huấn luyện vào thư mục model/
    model_path = "model/als_model"
    os.makedirs("model", exist_ok=True)
    model.write().overwrite().save(model_path)
    print(f"Đã lưu mô hình thành công tại: {model_path}")

    spark.stop()

if __name__ == "__main__":
    main()