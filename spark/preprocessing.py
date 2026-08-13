from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def main():
    # Thêm cấu hình để tránh lỗi Unresolved/UnsatisfiedLinkError trên Windows
    spark = SparkSession.builder \
        .appName("MovieLensETL") \
        .config("spark.driver.memory", "4g") \
        .config("spark.hadoop.io.nativeio.NativeIO$Windows", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    print("Đang đọc dữ liệu gốc...")
    # Lưu ý: Đảm bảo đường dẫn dataset nằm đúng thư mục thực tế của bạn
    ratings_df = spark.read.option("header", "true").option("inferSchema", "true").csv("dataset/ml-20m/ratings.csv")
    movies_df = spark.read.option("header", "true").option("inferSchema", "true").csv("dataset/ml-20m/movies.csv")

    print("Làm sạch dữ liệu (loại bỏ giá trị null)...")
    clean_ratings = ratings_df.dropna(subset=["userId", "movieId", "rating"])

    print("Ghi dữ liệu đã xử lý ra định dạng Parquet...")
    clean_ratings.write.mode("overwrite").parquet("dataset/processed/ratings_clean.parquet")
    
    print("ETL hoàn tất!")
    spark.stop()

if __name__ == "__main__":
    main()