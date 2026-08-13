# Big Data Movie Recommendation System

**PySpark ALS + Collaborative Filtering + Flask + Redis + Docker**

Hệ thống gợi ý phim sử dụng thuật toán **ALS (Alternating Least Squares)** trên Apache Spark, kết hợp Flask API và Redis Cache để cung cấp kết quả đề xuất phim.

## 1. Công nghệ sử dụng

| Công nghệ              | Mục đích                           |
| ---------------------- | ---------------------------------- |
| Python                 | Ngôn ngữ lập trình                 |
| PySpark / Apache Spark | Xử lý Big Data và Machine Learning |
| ALS                    | Collaborative Filtering            |
| Flask                  | Backend và REST API                |
| Redis                  | Cache kết quả recommendation       |
| Docker                 | Quản lý môi trường chạy            |
| MovieLens 20M          | Dataset                            |

## 2. Kiến trúc hệ thống

```text
MovieLens 20M
      |
      v
PySpark Preprocessing
      |
      v
Apache Spark ALS
      |
      v
Recommendation Model
      |
      v
Flask API <----> Redis Cache
      |
      v
Web Interface
```

## 3. Cấu trúc thư mục

```text
BIGDATA-RECOMMENDATION-SYSTEM/
├── dataset/
│   └── ml-20m/
├── docker/
│   └── docker-compose.yml
├── evaluation/
│   └── rmse_result.txt
├── flask/
│   ├── static/
│   ├── templates/
│   └── app.py
├── model/
│   └── als_model/
├── notebooks/
├── output/
│   ├── batch_recommendations/
│   └── metrics.json
├── results/
│   └── tuning_result.csv
├── screenshots/
├── spark/
│   ├── preprocessing.py
│   ├── train_als.py
│   ├── evaluation.py
│   ├── recommend.py
│   └── export_output.py
├── README.md
└── requirements.txt
```

## 4. Yêu cầu hệ thống

* Python 3.9+
* Java OpenJDK 8 hoặc 11
* Docker Desktop
* Git

Kiểm tra phiên bản:

```bash
python --version
java -version
docker --version
docker compose version
```

## 5. Cài đặt

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/BIGDATA-RECOMMENDATION-SYSTEM.git
cd BIGDATA-RECOMMENDATION-SYSTEM
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## 6. Khởi chạy

### Bước 1: Khởi động Docker

```bash
docker compose -f docker/docker-compose.yml up -d
```

Kiểm tra container:

```bash
docker ps
```

### Bước 2: Tiền xử lý dữ liệu

```bash
python spark/preprocessing.py
```

### Bước 3: Huấn luyện mô hình ALS

```bash
python spark/train_als.py
```

### Bước 4: Đánh giá mô hình

```bash
python spark/evaluation.py
```

### Bước 5: Kiểm tra recommendation

```bash
python spark/recommend.py --user_id 1
```

### Bước 6: Khởi chạy Flask

```bash
python flask/app.py
```

Truy cập:

```text
http://127.0.0.1:5000/
```

## 7. Recommendation Workflow

```text
User
 |
 v
Flask API
 |
 v
Redis Cache
 |
 +-- Cache HIT --> Return recommendations
 |
 +-- Cache MISS
        |
        v
    Spark ALS
        |
        v
 Recommendations
        |
        v
    Redis Cache
```

Request đầu tiên sẽ sử dụng Spark ALS để tính recommendation và lưu kết quả vào Redis.

Các request tiếp theo sẽ lấy kết quả trực tiếp từ Redis Cache để giảm thời gian phản hồi.

## 8. Đánh giá mô hình

Mô hình sử dụng **RMSE (Root Mean Square Error)** để đánh giá độ chính xác.

Kết quả được lưu tại:

```text
evaluation/rmse_result.txt
```

Kết quả tuning tham số:

```text
results/tuning_result.csv
```

## 9. Output

```text
output/
├── batch_recommendations/
└── metrics.json
```

## 10. Dataset

Dự án sử dụng **MovieLens 20M Dataset**, bao gồm dữ liệu về:

* Movies
* Users
* Ratings
* Tags

## 11. Project

**Big Data Movie Recommendation System**

PySpark ALS + Collaborative Filtering + Flask + Redis + Docker
