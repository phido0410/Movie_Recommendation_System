from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
import os
import json
from recommender import MovieRecommender
import pandas as pd

app = Flask(__name__)
app.config['TITLE'] = 'CineMatch'
app.config['PORT'] = 6789  # Port tùy chỉnh, không còn dùng 5000 mặc định

# Khởi tạo recommender system
recommender = MovieRecommender()

# Load poster mapping từ JSON file
def load_poster_mapping():
    try:
        poster_file = os.path.join(app.root_path, 'static/data/movie-poster.json')
        if os.path.exists(poster_file):
            with open(poster_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading poster mapping: {str(e)}")
        return {}

# Global poster mapping
poster_mapping = load_poster_mapping()

# Function to prepare dataframe for template (convert dates to strings and add poster paths)
def prepare_for_template(df, add_posters=True):
    df = df.copy()
    
    # Convert dates to strings
    if 'release_date' in df.columns:
        df['release_date'] = df['release_date'].astype(str)
    
    # Add poster paths for movies
    if add_posters:
        df['poster_path'] = df['original_title'].apply(
            lambda title: url_for('static', filename=f'posters/{poster_mapping.get(title, "no-poster.jpg")}')
        )
    else:
        # Use default poster for all movies
        df['poster_path'] = url_for('static', filename='images/no-poster.jpg')
    
    return df

def get_popular_and_most_voted():
    popular_movies = recommender.get_popular_movies(min_votes=1000, top_n=10)
    most_voted_movies = recommender.get_most_voted_movies(top_n=10)
    
    # Prepare for template with posters
    popular_movies = prepare_for_template(popular_movies, add_posters=True)
    most_voted_movies = prepare_for_template(most_voted_movies, add_posters=True)
    
    return popular_movies, most_voted_movies

@app.route('/')
def home():
    # Lấy danh sách phim phổ biến để hiển thị trên trang chủ
    popular_movies, most_voted_movies = get_popular_and_most_voted()
    
    return render_template(
        'index.html', 
        popular_movies=popular_movies.to_dict('records'),
        most_voted_movies=most_voted_movies.to_dict('records'),
        active_page='home'
    )

@app.route('/trending')
def trending():
    # Lấy danh sách phim xu hướng (kết hợp giữa đánh giá cao và nhiều lượt xem)
    trending_movies = recommender.get_trending_movies(top_n=20)
    
    # Chỉ thêm poster cho phim trong danh sách top, các phim khác dùng poster mặc định
    trending_movies = prepare_for_template(trending_movies, add_posters=False)
    
    # Áp dụng poster cho các phim có trong mapping
    for index, movie in trending_movies.iterrows():
        if movie['original_title'] in poster_mapping:
            trending_movies.at[index, 'poster_path'] = url_for('static', 
                                 filename=f'posters/{poster_mapping[movie["original_title"]]}')
    
    return render_template(
        'trending.html',
        trending_movies=trending_movies.to_dict('records'),
        active_page='trending'
    )

@app.route('/top-rated')
def top_rated():
    # Lấy danh sách phim đánh giá cao
    top_rated_movies = recommender.get_top_rated_movies(top_n=20)
    
    # Chỉ thêm poster cho phim trong danh sách top, các phim khác dùng poster mặc định
    top_rated_movies = prepare_for_template(top_rated_movies, add_posters=False)
    
    # Áp dụng poster cho các phim có trong mapping
    for index, movie in top_rated_movies.iterrows():
        if movie['original_title'] in poster_mapping:
            top_rated_movies.at[index, 'poster_path'] = url_for('static', 
                                 filename=f'posters/{poster_mapping[movie["original_title"]]}')
    
    return render_template(
        'top_rated.html',
        top_rated_movies=top_rated_movies.to_dict('records'),
        active_page='top_rated'
    )

@app.route('/about')
def about():
    # Lấy thống kê dữ liệu phim
    stats = recommender.get_stats()
    
    return render_template(
        'about.html',
        stats=stats,
        active_page='about'
    )

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title', '')
    
    if not movie_title:
        popular_movies, most_voted_movies = get_popular_and_most_voted()
        return render_template(
            'index.html', 
            error="Vui lòng nhập tên phim",
            popular_movies=popular_movies.to_dict('records'),
            most_voted_movies=most_voted_movies.to_dict('records'),
            active_page='home'
        )
    
    # Lấy đề xuất phim
    recommended_movies, similarity_scores = recommender.get_recommendations(movie_title)
    
    if recommended_movies is None:
        popular_movies, most_voted_movies = get_popular_and_most_voted()
        return render_template(
            'index.html', 
            error=f"Không tìm thấy phim: {movie_title}",
            popular_movies=popular_movies.to_dict('records'),
            most_voted_movies=most_voted_movies.to_dict('records'),
            active_page='home'
        )
    
    # Chuẩn bị dữ liệu cho template - mặc định dùng poster no-poster.jpg
    recommended_movies = prepare_for_template(recommended_movies, add_posters=False)
    
    # Áp dụng poster cho các phim có trong mapping
    for index, movie in recommended_movies.iterrows():
        if movie['original_title'] in poster_mapping:
            recommended_movies.at[index, 'poster_path'] = url_for('static', 
                                 filename=f'posters/{poster_mapping[movie["original_title"]]}')
    
    # Lấy thông tin chi tiết về phim được chọn
    try:
        movie_df = recommender.movies_df[recommender.movies_df['original_title'] == movie_title]
        if len(movie_df) > 0:
            # Chuyển từ DataFrame row thành dictionary
            movie_df = prepare_for_template(movie_df, add_posters=False)
            selected_movie = movie_df.iloc[0].to_dict()
            
            # Thêm poster cho phim được chọn nếu có trong danh sách
            if movie_title in poster_mapping:
                selected_movie['poster_path'] = url_for('static', 
                                           filename=f'posters/{poster_mapping[movie_title]}')
        else:
            selected_movie = None
    except Exception as e:
        print(f"Error getting movie details: {str(e)}")
        selected_movie = None
    
    # Chuyển đổi các đề xuất thành list of dicts và thêm điểm tương đồng
    recommendations = []
    for i, movie in enumerate(recommended_movies.to_dict('records')):
        movie['similarity'] = round(similarity_scores[i] * 100, 2)
        recommendations.append(movie)
    
    return render_template(
        'recommendations.html', 
        movie_title=movie_title,
        selected_movie=selected_movie,
        recommendations=recommendations,
        active_page=None
    )

# Thêm route mới để xử lý tìm kiếm phim
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    
    if not query:
        return jsonify([])
    
    # Tìm các phim phù hợp với truy vấn
    matching_movies = recommender.search_movies(query)
    
    # Chuẩn bị dữ liệu với poster
    matching_movies = prepare_for_template(matching_movies, add_posters=False)
    
    # Thêm poster cho phim được tìm thấy có trong mapping
    for index, movie in matching_movies.iterrows():
        if movie['original_title'] in poster_mapping:
            matching_movies.at[index, 'poster_path'] = url_for('static', 
                                 filename=f'posters/{poster_mapping[movie["original_title"]]}')
    
    # Trả về kết quả dưới dạng JSON
    return jsonify(matching_movies.to_dict('records'))

# Serving favicon.ico
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    # Chạy app với port tùy chỉnh
    app.run(debug=True, host='0.0.0.0', port=app.config['PORT'])