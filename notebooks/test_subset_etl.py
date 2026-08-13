import findspark
findspark.init()

from pyspark.sql import SparkSession

def main():
    # Khởi tạo Spark Session với tối ưu hóa cơ bản
    spark = SparkSession.builder \
        .appName("MovieLensETLSubset") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    print("Đang đọc tập dữ liệu ratings.csv từ thư mục dataset...")
    # Cập nhật đường dẫn chuẩn trỏ tới thư mục ml-20m
    ratings_df = spark.read.option("header", "true").option("inferSchema", "true").csv("dataset/ml-20m/ratings.csv")

    # Kiểm thử trên subset 1% dữ liệu (fraction = 0.01) theo đúng hướng dẫn đề tài
    print("Đang cắt subset 1% dữ liệu để kiểm thử...")
    subset_df = ratings_df.sample(withReplacement=False, fraction=0.01, seed=42)

    # Hiển thị thông tin kiểm tra
    print(f"Tổng số dòng trong bản full (dự kiến khoảng 20 triệu): {ratings_df.count()}")
    print(f"Tổng số dòng trong subset 1%: {subset_df.count()}")
    
    subset_df.show(5)

    spark.stop()

if __name__ == "__main__":
    main()