from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

def main():
    print("Đang khởi tạo Spark Session...")
    spark = SparkSession.builder \
        .appName("RecommendationOutput") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    print("Đang đọc dữ liệu từ dataset/ml-20m/ratings.csv...")
    ratings_df = spark.read.option("header", "true").option("inferSchema", "true").csv("dataset/ml-20m/ratings.csv")

    print("Đang huấn luyện mô hình ALS để tạo gợi ý...")
    als = ALS(
        maxIter=5, 
        regParam=0.01, 
        rank=10, 
        userCol="userId", 
        itemCol="movieId", 
        ratingCol="rating", 
        coldStartStrategy="drop"
    )
    model = als.fit(ratings_df)

    print("Đang tính toán top 10 phim gợi ý cho người dùng...")
    users = ratings_df.select(als.getUserCol()).distinct().limit(5)
    recommendations = model.recommendForUserSubset(users, 10)
    
    print("Hoàn tất! Kết quả gợi ý:")
    recommendations.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()