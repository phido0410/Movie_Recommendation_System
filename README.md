
# 🎬 CineMatch - Hệ thống đề xuất phim dựa trên nội dung

<p align="center">
  <img src="Pic/picreadme.png" alt="CineMatch Logo" width="200"/>
</p>

## 📌 Giới thiệu

**CineMatch** là một hệ thống đề xuất phim thông minh sử dụng các thuật toán trí tuệ nhân tạo để phân tích nội dung phim và đưa ra những đề xuất phù hợp nhất với sở thích của bạn.

---

## 🚀 Tính năng chính

- 🔍 **Tìm kiếm phim theo tên** với gợi ý thông minh
- 🎞️ **Đề xuất phim tương tự** dựa trên nội dung của phim đã chọn
- 🌟 **Danh sách phim nổi bật** được đánh giá cao
- 📈 **Phim xu hướng** đang được quan tâm
- 💻 **Giao diện người dùng hiện đại** phong cách Netflix, responsive trên mọi thiết bị

---

## 🧠 Công nghệ sử dụng

### 🔧 Back-end
- **Python**: Ngôn ngữ chính
- **Flask**: Web framework
- **Pandas, NumPy**: Xử lý dữ liệu
- **Scikit-learn**: Thuật toán TF-IDF & Cosine Similarity

### 🎨 Front-end
- **HTML/CSS**, **JavaScript**
- **Bootstrap 5**, **Font Awesome**, **Google Fonts**

---


## 📁 Cấu trúc dự án

```
Movie_Recommendation_System/
├── .gitattributes                         # Cấu hình Git attributes
├── Chuong5_RecommenderSystem.pdf          # Tài liệu lý thuyết về hệ thống đề xuất
├── data_preprocessing.ipynb               # Notebook xử lý dữ liệu
├── requirements.txt                       # Danh sách thư viện cần thiết
├── Data/                                  # Thư mục chứa dữ liệu gốc
│   ├── Data_Movies_ok.csv                 # Dữ liệu phim đã xử lý
│   ├── Data_Movies.csv                    # Dữ liệu phim gốc
│   └── Data_VN_2021.xlsx                  # Dữ liệu bổ sung về các tỉnh Việt Nam
├── movie_recomender/                      # Thư mục chứa mã nguồn chính của ứng dụng
│   ├── .gitignore                         # Cấu hình Git ignore
│   ├── app.py                             # Mã nguồn chính Flask application
│   ├── matrix_loader.py                   # Module tải ma trận tương đồng
│   ├── recommender.py                     # Module xử lý đề xuất phim
│   ├── __pycache__/                       # Thư mục cache của Python
│   │   └── recommender.cpython-312.pyc    # Tệp pyc đã biên dịch
│   ├── data/                              # Thư mục dữ liệu cho ứng dụng
│   │   ├── Data_Movies_ok.csv             # Bản sao của dữ liệu phim đã xử lý
│   │   └── similarity_matrix.npz          # Ma trận tương đồng đã được tính toán
│   ├── static/                            # Thư mục tĩnh cho web
│   │   ├── favicon.ico                    # Icon cho website
│   │   ├── style.css                      # Tệp CSS chính
│   │   ├── data/                          # Dữ liệu tĩnh
│   │   ├── images/                        # Hình ảnh chung
│   │   └── posters/                       # Poster phim
│   └── templates/                         # Thư mục chứa các template HTML
│       ├── about.html                     # Trang giới thiệu
│       ├── index.html                     # Trang chủ
│       ├── recommendations.html           # Trang hiển thị kết quả đề xuất
│       ├── top_rated.html                 # Trang phim được đánh giá cao
│       └── trending.html                  # Trang phim xu hướng
└── Pic/                                   # Thư mục chứa hình ảnh minh họa
    ├── cosine_sim.png                     # Ảnh minh họa độ tương đồng cosine
    ├── pic_cosine.png                     # Ảnh minh họa về cosine similarity
    ├── pic0.jpg                           # Các ảnh minh họa khác
    ├── pic1.png
    ├── pic2.png
    ├── pic4.jpg
    └── pic5.jpg
```
---

## 🧩 Quy trình hoạt động

1. **Tiền xử lý dữ liệu**
   - Làm sạch, loại bỏ giá trị thiếu, chuẩn hóa
2. **Xây dựng mô hình**
   - Vector hóa nội dung bằng **TF-IDF**
   - Tính **Cosine Similarity** giữa các phim
   - Lưu ma trận tương đồng để tối ưu hiệu suất
3. **Đề xuất**
   - Trả về danh sách phim có nội dung gần nhất với phim được chọn

---

## ⚙️ Cài đặt và chạy ứng dụng

### ✅ Yêu cầu
- Python 3.8+
- Pip

### 🔨 Các bước triển khai

```bash
# 1. Clone repo
git clone <repository-url>
cd Movie_Recommendation_System

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy ứng dụng
cd movie_recomender
python app.py
```

➡️ Mở trình duyệt và truy cập: `http://localhost:6789`

---

## 🌟 Tính năng nổi bật

### 1. Hệ thống đề xuất nội dung
- Sử dụng **TF-IDF** và **Cosine Similarity** để tìm phim tương tự

### 2. Tìm kiếm thông minh
- Tìm kiếm realtime với hình ảnh minh họa rõ ràng

### 3. Hiệu suất cao
- Dữ liệu lưu dưới dạng sparse matrix
- Tải ma trận từ Google Drive nếu thiếu
- Dùng caching để tăng tốc phản hồi

### 4. UI hiện đại
- Thiết kế kiểu **Netflix**
- Sử dụng AOS cho hiệu ứng mượt
- Responsive toàn diện

---

## 📊 Thống kê dữ liệu

| Thông tin             | Giá trị              |
|----------------------|----------------------|
| Số lượng phim        | 11756                |
| Giai đoạn phát hành  | 1085 - 2020          |
| Điểm đánh giá TB     | 6.3/10               |
| Lượt đánh giá TB     | 396/phim             |

---

## 👤 Tác giả

- **Đỗ Ngọc Phi** - MSSV: 2221050848
- **Nguyễn Minh Quân** - MSSV: 2221050125
- **Đào Anh Tú** - 2221050231
- **GVHD**: Thầy Đặng Văn Nam, Cô Dương Thị Hiền Thanh  
  _Khoa CNTT - Trường ĐH Mỏ - Địa chất_

---

## 📚 Tài liệu tham khảo

- Bài giảng "Machine Learning - Chương 5: Recommender Systems"
- Tài liệu chính thức của Scikit-learn
- Nguồn học thuật về TF-IDF & Cosine Similarity

---

## 📝 Ghi chú cuối

Dự án **CineMatch** là ví dụ tiêu biểu về việc tích hợp học máy và phát triển web. Các yếu tố như cấu trúc rõ ràng, hiệu suất tối ưu, giao diện thân thiện giúp dự án không chỉ tốt về mặt kỹ thuật mà còn hoàn thiện về trải nghiệm người dùng.

---

⭐ Nếu bạn thấy dự án hữu ích, hãy ⭐ trên GitHub nhé!
