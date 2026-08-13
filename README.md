# Big Data Movie Recommendation System (PySpark ALS + Flask + Redis)

## Yêu cầu hệ thống (Prerequisites)

Trước khi khởi chạy dự án, máy tính cần cài đặt sẵn:
1. **Python 3.9+**
2. **Docker Desktop** (Đã mở và đang chạy background)
3. **Java OpenJDK 8 hoặc 11** (Để hỗ trợ môi trường PySpark local)

I. Kiến trúc hệ thống (System Architecture)
┌─────────────────────────────────────────────────────────┐
│                      Caching Layer                      │
│               (Redis - Tối ưu hiệu năng)                │
└─────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│          (Flask API + Frontend Glassmorphism)           │
└─────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────┐
│                   Big Data & ML Layer                   │
│      (Apache Spark ALS - Collaborative Filtering)       │
└─────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────┐
│                      Data Layer                         │
│          (MovieLens 20M Dataset - Raw Data)             │
└─────────────────────────────────────────────────────────┘

II. Cấu trúc thư mục dự án
BIGDATA-RECOMMENDATION-SYSTEM/
├── dataset/             # Dữ liệu đầu vào MovieLens (ml-20m)
│   └── ml-20m/          # ratings.csv, movies.csv, tags.csv, ...
├── docker/              # Cấu hình Docker Compose (Spark, Kafka, Redis)
│   └── docker-compose.yml
├── evaluation/          # Kết quả đánh giá mô hình
│   └── rmse_result.txt
├── flask/               # Code Web Application & API
│   ├── static/          # File tĩnh (style.css, background image)
│   ├── templates/       # Giao diện HTML (index.html)
│   └── app.py           # Flask Server chính kết nối Spark & Redis
├── model/               # Lưu mô hình Spark ALS đã huấn luyện
│   └── als_model/       # itemFactors, userFactors, metadata
├── notebooks/           # Code thử nghiệm / ETL subset
├── output/              # Dữ liệu đầu ra & Metrics
│   ├── batch_recommendations/
│   └── metrics.json
├── results/             # Kết quả tinh chỉnh tham số (Hyperparameter Tuning)
│   └── tuning_result.csv
├── screenshots/         # Ảnh minh chứng chạy hệ thống
├── spark/               # Các script xử lý PySpark
│   ├── evaluation.py    # Đánh giá độ chính xác (RMSE)
│   ├── export_output.py # Script xuất dữ liệu gợi ý & metrics ra file
│   ├── preprocessing.py # Tiền xử lý dữ liệu
│   ├── recommend.py     # Script test dự đoán trực tiếp
│   └── train_als.py     # Huấn luyện mô hình ALS
├── README.md            # Tài liệu hướng dẫn dự án
└── requirements.txt     # Danh sách các thư viện Python cần cài đặt

III. Hướng dẫn cài đặt & Khởi chạy (Quick Start)

Bước 1: Khởi tạo hạ tầng Docker Container
Khởi chạy đồng thời 3 container Spark Master, Apache Kafka và Redis Cache:
                    
                    docker-compose up -d

Bước 2: Cài đặt thư viện Python
                  
                    pip install -r requirements.txt

Bước 3: Tiền xử lý dữ liệu (ETL / Preprocessing)
Đọc và làm sạch 20 triệu lượt đánh giá từ tập dữ liệu MovieLens:
                   
                    python spark/preprocessing.py

Bước 4: Huấn luyện mô hình ALS
Phân rã ma trận người dùng - bộ phim và lưu trained model vào thư mục model/:
                    
                    python spark/train_als.py

Bước 5: Đánh giá độ chính xác mô hình (RMSE)
                   
                    python spark/evaluation.py

Bước 6: Kiểm tra gợi ý trên Terminal
Chạy thử tính toán gợi ý trực tiếp cho một user_id bất kỳ (ví dụ: User 1):
                  
                    python spark/recommend.py --user_id 1

Bước 7: Khởi chạy Web Backend (Flask API)
                    
                    python flask/app.py

TRUY CẬP GIAO DIỆN WEB TẠI: http://127.0.0.1:5000/

IV. Cơ chế vận hành Đề xuất (Recommendation Workflow)
Lần truy vấn đầu tiên (First Request): Khi nhận user_id mới từ phía Web, Web Server gọi trực tiếp Mô hình Spark ALS để tính toán ma trận gợi ý theo thời gian thực.

Caching: Kết quả tính toán sẽ được tự động lưu vết vào Redis Cache.

Các lần truy vấn tiếp theo (Subsequent Requests): Flask API sẽ ưu tiên lấy kết quả trực tiếp từ Redis Cache, giúp thời gian phản hồi giao diện đạt mức tiệm cận 0ms.