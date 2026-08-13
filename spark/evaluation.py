from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import csv
import os

def main():
    spark = SparkSession.builder \
        .appName("ALSTuningAndEvaluation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.hadoop.io.nativeio.NativeIO$Windows", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    print("Đang đọc dữ liệu đã làm sạch từ định dạng Parquet...")
    ratings_df = spark.read.parquet("dataset/processed/ratings_clean.parquet")
    
    # Chia dữ liệu 80% train, 20% test
    (train_df, test_df) = ratings_df.randomSplit([0.8, 0.2], seed=42)

    # Thử nghiệm các cấu hình tuning (Rank và Regularization)
    param_grid = [
        {"rank": 10, "regParam": 0.1},
        {"rank": 20, "regParam": 0.05},
        {"rank": 30, "regParam": 0.1}
    ]

    os.makedirs("results", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    results = []
    best_rmse = float('inf')
    best_params = None

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    for params in param_grid:
        print(f"Đang huấn luyện với rank={params['rank']}, regParam={params['regParam']}...")
        als = ALS(
            maxIter=10,
            rank=params['rank'],
            regParam=params['regParam'],
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop"
        )
        model = als.fit(train_df)
        predictions = model.transform(test_df)
        rmse = evaluator.evaluate(predictions)
        print(f"-> RMSE: {rmse}")

        results.append([params['rank'], params['regParam'], rmse])

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    # Lưu kết quả tuning vào file CSV
    with open("results/tuning_result.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "regParam", "RMSE"])
        writer.writerows(results)

    # Lưu kết quả RMSE tốt nhất
    with open("evaluation/rmse_result.txt", mode="w", encoding="utf-8") as f:
        f.write(f"Best Rank: {best_params['rank']}\n")
        f.write(f"Best RegParam: {best_params['regParam']}\n")
        f.write(f"Best RMSE: {best_rmse}\n")

    print("Tuning và đánh giá hoàn tất! Đã lưu kết quả vào thư mục results/ và evaluation/.")
    spark.stop()

if __name__ == "__main__":
    main()