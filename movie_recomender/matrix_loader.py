import os
import numpy as np
import requests
import io
import gdown

def load_similarity_matrix():
    # Kiểm tra file có tồn tại local không (chạy trong môi trường phát triển)
    local_path = 'similarity_matrix.npz'
    if os.path.exists(local_path):
        print("Loading similarity matrix from local file...")
        data = np.load(local_path, allow_pickle=True)
        # Trả về ma trận từ key phù hợp
        if 'array' in data:
            return data['array']
        else:
            # Lấy key đầu tiên nếu không rõ key
            return data[list(data.keys())[0]]
    
    # Nếu không có file local, tải từ Google Drive
    print("Loading similarity matrix from Google Drive...")
    
    # ID của file trên Google Drive (lấy từ link chia sẻ của bạn)
    file_id = '1q7EYa332RknMmYqTcBsn_5eH_VjCLbMm'
    
    # Phương pháp 1: Sử dụng gdown (khuyên dùng)
    try:
        # Đường dẫn file tạm thời
        temp_path = '/tmp/similarity_matrix.npz'
        
        # Tải file từ Google Drive
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, temp_path, quiet=False)
        
        # Đọc ma trận từ file tạm
        data = np.load(temp_path, allow_pickle=True)
        
        # Xác định key trong file npz
        if 'array' in data:
            return data['array']
        else:
            # Lấy key đầu tiên nếu không rõ key
            return data[list(data.keys())[0]]
    
    # Phương pháp 2: Sử dụng requests nếu phương pháp 1 thất bại
    except Exception as e:
        print(f"Error using gdown: {str(e)}")
        print("Trying alternative method...")
        
        try:
            # Link trực tiếp
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(direct_url)
            
            if response.status_code == 200:
                with io.BytesIO(response.content) as f:
                    data = np.load(f, allow_pickle=True)
                    
                    # Xác định key trong file npz
                    if 'array' in data:
                        return data['array']
                    else:
                        # Lấy key đầu tiên nếu không rõ key
                        return data[list(data.keys())[0]]
            else:
                print(f"Failed to download file: {response.status_code}")
                # Trả về mảng rỗng trong trường hợp lỗi
                return np.array([])
                
        except Exception as e:
            print(f"Error downloading similarity matrix: {str(e)}")
            # Trả về mảng rỗng trong trường hợp lỗi
            return np.array([])