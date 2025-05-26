import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import re
from scipy import sparse
import requests
import io
import gdown
import gc  # Thêm thư viện garbage collector

class MovieRecommender:
    def __init__(self, csv_path='data/Data_Movies_ok.csv', similarity_matrix_path='data/similarity_matrix.npz'):
        self.csv_path = csv_path
        self.similarity_matrix_path = similarity_matrix_path
        self.gdrive_id = '1q7EYa332RknMmYqTcBsn_5eH_VjCLbMm'  # ID của file trên Google Drive
        
        # Đọc dữ liệu phim (giới hạn số lượng tùy môi trường)
        self.movies_df = self._load_data()
        
        # Tính toán/nạp ma trận tương đồng
        self.similarity_matrix = self._load_or_compute_similarity_matrix()
        
        # Kiểm tra loại ma trận
        self.is_sparse = isinstance(self.similarity_matrix, sparse.spmatrix)
        print(f"Using {'sparse' if self.is_sparse else 'dense'} similarity matrix")
    
    def _load_data(self):
        """Đọc và chuẩn hóa dữ liệu phim từ CSV"""
        try:
            # Đọc dữ liệu từ CSV
            df = pd.read_csv(self.csv_path)
            
            # Giới hạn số lượng phim khi chạy trên Render để giảm kích thước bộ nhớ
            if os.environ.get('RENDER', '0') == '1':
                # Giảm xuống 2500 phim để tiết kiệm RAM hơn nữa
                df = df.sort_values(by='vote_count', ascending=False).head(2500)
                print(f"Limited dataset to 2500 movies for Render deployment")
                
                # Gọi garbage collector để giải phóng bộ nhớ
                gc.collect()
            
            # Chuẩn hóa dữ liệu
            df['overview'] = df['overview'].fillna('')  # Thay thế NaN bằng chuỗi trống
            
            # Chuyển đổi các cột date thành datetime
            if 'release_date' in df.columns:
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            
            # Đảm bảo có đầy đủ các cột cần thiết
            required_columns = ['id', 'original_title', 'overview', 'vote_average', 'vote_count']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Missing required column '{col}' in dataset")
            
            # Thêm poster_path nếu không có
            if 'poster_path' not in df.columns:
                df['poster_path'] = "/static/images/no-poster.jpg"
            else:
                # Chuẩn hóa poster path
                df['poster_path'] = df['poster_path'].apply(
                    lambda x: f"https://image.tmdb.org/t/p/w500{x}" if pd.notna(x) and x else "/static/images/no-poster.jpg"
                )
            
            print(f"Loaded {len(df)} movies from {self.csv_path}")
            return df
            
        except Exception as e:
            print(f"Error loading movie data: {str(e)}")
            # Trả về DataFrame rỗng nếu có lỗi
            return pd.DataFrame(columns=['id', 'original_title', 'overview', 'vote_average', 'vote_count'])

    def _compute_similarity_matrix(self):
        """Tính toán ma trận tương đồng dựa trên overview của phim"""
        start_time = time.time()
        print("Computing similarity matrix...")
        
        # Tạo TF-IDF vectorizer và tính vectors
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['overview'])
        
        # Giải phóng bộ nhớ
        gc.collect()
        
        # Tính ma trận tương đồng cosine
        if os.environ.get('RENDER', '0') == '1':
            # Tính trực tiếp dạng sparse để tiết kiệm bộ nhớ
            print("Computing sparse similarity matrix for Render...")
            from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
            similarity = sk_cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)
            print(f"Sparse similarity matrix shape: {similarity.shape}")
        else:
            # Tính dạng dense cho môi trường local
            similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Giải phóng bộ nhớ
        del tfidf_matrix
        gc.collect()
        
        print(f"Similarity matrix computation completed in {time.time() - start_time:.2f} seconds")
        return similarity
    
    def _save_similarity_matrix(self, similarity_matrix):
        """Lưu ma trận tương đồng vào file để tái sử dụng"""
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.similarity_matrix_path), exist_ok=True)
        
        # Chuyển về dạng sparse nếu chưa phải
        if not isinstance(similarity_matrix, sparse.spmatrix):
            sparse_matrix = sparse.csr_matrix(similarity_matrix)
        else:
            sparse_matrix = similarity_matrix
            
        # Lưu ma trận dưới dạng sparse để tiết kiệm dung lượng
        sparse.save_npz(self.similarity_matrix_path, sparse_matrix)
        print(f"Similarity matrix saved to {self.similarity_matrix_path}")
    
    def _download_and_process_in_chunks(self, url, temp_file):
        """Tải và xử lý file lớn theo từng phần để tiết kiệm bộ nhớ"""
        try:
            # Sử dụng requests với streaming để tải từng phần nhỏ
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                print(f"Failed to download file: {response.status_code}")
                return None
            
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(temp_file), exist_ok=True) if os.path.dirname(temp_file) else None
            
            # Ghi từng chunk vào file
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            
            return temp_file
        except Exception as e:
            print(f"Error downloading in chunks: {str(e)}")
            return None
    
    def _load_from_gdrive(self, keep_sparse=False):
        """Tải ma trận tương đồng từ Google Drive"""
        print("Attempting to download similarity matrix from Google Drive...")
        
        # Sử dụng phương pháp tải theo chunk
        try:
            # URL trực tiếp (có thể cần sửa nếu Google Drive yêu cầu xác thực)
            direct_url = f"https://drive.google.com/uc?export=download&id={self.gdrive_id}"
            temp_file = 'temp_similarity_matrix.npz'
            
            # Tải file theo từng phần
            downloaded_file = self._download_and_process_in_chunks(direct_url, temp_file)
            
            if downloaded_file and os.path.exists(downloaded_file):
                # Đọc ma trận từ file tạm
                sparse_matrix = sparse.load_npz(downloaded_file)
                
                # Copy file tạm vào vị trí chính thức
                os.makedirs(os.path.dirname(self.similarity_matrix_path), exist_ok=True)
                import shutil
                shutil.copy(downloaded_file, self.similarity_matrix_path)
                
                # Xóa file tạm để giải phóng dung lượng
                os.remove(downloaded_file)
                gc.collect()  # Gọi garbage collector
                
                print("Successfully loaded sparse similarity matrix")
                return sparse_matrix  # Luôn giữ sparse khi chạy trên Render
                
        except Exception as e:
            print(f"Error downloading similarity matrix: {str(e)}")
            
            # Thử phương pháp với gdown nếu phương pháp chunk không hoạt động
            try:
                temp_file = 'temp_similarity_matrix.npz'
                url = f'https://drive.google.com/uc?id={self.gdrive_id}'
                
                # Tải file từ Google Drive
                gdown.download(url, temp_file, quiet=False)
                
                if os.path.exists(temp_file):
                    # Đọc ma trận từ file tạm
                    sparse_matrix = sparse.load_npz(temp_file)
                    
                    # Xóa file tạm ngay lập tức để giải phóng dung lượng
                    os.remove(temp_file)
                    gc.collect()
                    
                    print("Successfully loaded sparse similarity matrix using gdown")
                    return sparse_matrix
            except Exception as e:
                print(f"Error using gdown: {str(e)}")
        
        # Nếu tất cả phương pháp đều thất bại, trả về None
        print("Failed to download similarity matrix.")
        return None
    
    def _load_or_compute_similarity_matrix(self):
        """Kiểm tra nếu ma trận đã tồn tại thì load, nếu không thì tính và lưu"""
        # Khi chạy trên Render, luôn giữ ở dạng sparse
        render_env = os.environ.get('RENDER', '0') == '1'
        
        # Trên Render, tính toán mới ma trận tương đồng thay vì tải
        if render_env:
            print("Computing new similarity matrix for Render environment...")
            similarity = self._compute_similarity_matrix()
            return similarity  # _compute_similarity_matrix đã đảm bảo trả về sparse cho Render
        
        # Thử load ma trận từ local trước
        if os.path.exists(self.similarity_matrix_path):
            try:
                sparse_matrix = sparse.load_npz(self.similarity_matrix_path)
                
                # Kiểm tra kích thước
                if sparse_matrix.shape[0] == len(self.movies_df):
                    if render_env:
                        # Giữ nguyên sparse matrix trên Render
                        return sparse_matrix
                    else:
                        # Chuyển về dense trên môi trường local
                        return sparse_matrix.toarray()
                else:
                    print(f"Shape mismatch: similarity {sparse_matrix.shape[0]}, movies {len(self.movies_df)}")
            except Exception as e:
                print(f"Error loading local similarity matrix: {str(e)}")
        
        # Tính toán lại
        similarity = self._compute_similarity_matrix()
        self._save_similarity_matrix(similarity)
        
        # Không cần chuyển đổi vì _compute_similarity_matrix đã trả về đúng định dạng
        return similarity
    
    def get_recommendations(self, movie_title, top_n=10):
        """
        Tìm các phim tương tự dựa trên tên phim
        Trả về danh sách top_n phim có điểm tương đồng cao nhất
        """
        # Kiểm tra xem phim có trong dataset không
        movie_idx = self.movies_df[self.movies_df['original_title'] == movie_title].index
        
        if len(movie_idx) == 0:
            # Tìm phim gần giống nhất nếu không có kết quả chính xác
            similar_titles = self.search_movies(movie_title, limit=1)
            if len(similar_titles) > 0:
                movie_title = similar_titles.iloc[0]['original_title']
                movie_idx = self.movies_df[self.movies_df['original_title'] == movie_title].index
            else:
                print(f"Movie '{movie_title}' not found in dataset")
                return None, None
        
        # Lấy điểm tương đồng của phim này với tất cả các phim khác
        movie_idx = movie_idx[0]
        
        # Xử lý khác nhau tùy thuộc vào loại ma trận (sparse hoặc dense)
        if self.is_sparse:
            # Tối ưu hóa xử lý sparse matrix
            similarity_row = self.similarity_matrix[movie_idx]
            
            # Chuyển CSR matrix thành đôi (index, giá trị) và sắp xếp theo giá trị
            similarity_scores = sorted(
                [(i, similarity_row[0, i]) for i in similarity_row.indices], 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Thêm vào các phim còn thiếu nếu cần
            if len(similarity_scores) < top_n + 1:
                # Tìm các indices chưa có
                existing_indices = set(similarity_row.indices)
                missing_indices = [i for i in range(len(self.movies_df)) if i not in existing_indices and i != movie_idx]
                
                # Thêm vào các indices còn thiếu với điểm tương đồng bằng 0
                similarity_scores.extend([(i, 0.0) for i in missing_indices])
                
                # Sắp xếp lại
                similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        else:
            # Xử lý như bình thường nếu là dense matrix
            similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Lấy top_n+1 (bỏ qua phim đầu tiên vì nó chính là phim đang tìm)
        top_similar = similarity_scores[1:top_n+1] if movie_idx == similarity_scores[0][0] else similarity_scores[:top_n]
        
        # Lấy indices của các phim tương tự
        movie_indices = [i[0] for i in top_similar]
        # Lấy điểm tương đồng
        similarity_values = [float(i[1]) for i in top_similar]
        
        # Trả về DataFrame chứa thông tin của các phim tương tự
        return self.movies_df.iloc[movie_indices], similarity_values
    
    # [Các phương thức khác giữ nguyên]
    def search_movies(self, query, limit=10):
        """
        Tìm kiếm phim theo tên sử dụng các kỹ thuật fuzzy matching
        """
        if not query or len(query) < 2:
            return pd.DataFrame()
        
        # Chuyển query thành lowercase và loại bỏ các ký tự đặc biệt
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        
        # Tạo trường title_lower để so sánh không phân biệt chữ hoa/thường
        self.movies_df['title_lower'] = self.movies_df['original_title'].str.lower()
        
        # Tìm các phim có title chứa query
        results = self.movies_df[self.movies_df['title_lower'].str.contains(query, na=False)]
        
        # Sắp xếp kết quả: ưu tiên những phim có title bắt đầu bằng query
        starts_with = results[results['title_lower'].str.startswith(query, na=False)]
        contains = results[~results['title_lower'].str.startswith(query, na=False)]
        
        # Ghép các kết quả lại, ưu tiên popularity và những phim có nhiều vote hơn
        results = pd.concat([starts_with, contains])
        results = results.sort_values(by=['vote_count', 'vote_average'], ascending=False)
        
        # Drop cột tạm
        results = results.drop(columns=['title_lower'])
        
        # Trả về tối đa limit phim
        return results.head(limit)
    
    def get_popular_movies(self, min_votes=1000, top_n=10):
        """Lấy danh sách phim được đánh giá cao nhất (có ít nhất min_votes lượt đánh giá)"""
        popular = self.movies_df[self.movies_df['vote_count'] >= min_votes]
        return popular.sort_values('vote_average', ascending=False).head(top_n)
    
    def get_most_voted_movies(self, top_n=10):
        """Lấy danh sách phim có nhiều lượt đánh giá nhất"""
        return self.movies_df.sort_values('vote_count', ascending=False).head(top_n)

    def get_trending_movies(self, min_votes=500, top_n=20):
        """Lấy danh sách phim xu hướng dựa trên đánh giá cao và lượt vote nhiều"""
        # Tạo công thức tính "xu hướng" kết hợp giữa vote_average và vote_count
        df = self.movies_df.copy()
        df['trending_score'] = df['vote_average'] * np.log1p(df['vote_count'])
        trending = df[df['vote_count'] >= min_votes]
        return trending.sort_values('trending_score', ascending=False).head(top_n)
    
    def get_top_rated_movies(self, min_votes=1000, top_n=20):
        """Lấy danh sách phim đánh giá cao, mở rộng hơn get_popular_movies"""
        return self.get_popular_movies(min_votes=min_votes, top_n=top_n)
    
    def get_stats(self):
        """Lấy thống kê cơ bản về dữ liệu phim"""
        stats = {
            'total_movies': len(self.movies_df),
            'avg_rating': round(self.movies_df['vote_average'].mean(), 1),
            'avg_votes': f"{int(self.movies_df['vote_count'].mean()):,}",
            'oldest_movie': self.movies_df['release_date'].min().year if 'release_date' in self.movies_df.columns else 'N/A',
            'newest_movie': self.movies_df['release_date'].max().year if 'release_date' in self.movies_df.columns else 'N/A',
        }
        return stats