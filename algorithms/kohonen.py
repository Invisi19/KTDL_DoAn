import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Bước 1: Xử lý giá trị thiếu
    data = data.dropna()  # Xóa các hàng có giá trị thiếu. Hoặc có thể thay thế giá trị thiếu nếu cần.

    # Bước 2: Mã hóa các giá trị chuỗi thành số
    for col in data.select_dtypes(include=['object']):  # Chỉ áp dụng cho cột kiểu 'object'
        data[col] = data[col].astype('category').cat.codes

    # Bước 3: Chuẩn hóa dữ liệu (scaling) để tất cả các đặc trưng có giá trị trong cùng một phạm vi
    scaler = StandardScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data_normalized

def kohonen_algorithm(data, x_size=10, y_size=10, max_iter=100, learning_rate=0.5, sigma=1.0):
    # Chuẩn hóa dữ liệu
    data = preprocess_data(data)
    data_array = data.to_numpy()

    # Khởi tạo trọng số nơ-ron ngẫu nhiên
    n_samples, n_features = data_array.shape
    weights = np.random.random((x_size, y_size, n_features))

    # Hàm tính khoảng cách Euclid giữa các nơ-ron và điểm dữ liệu
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    # Huấn luyện mạng Kohonen
    for iteration in range(max_iter):
        for sample in data_array:
            # Tính toán khoảng cách từ mỗi điểm dữ liệu đến các nơ-ron
            distances = np.linalg.norm(weights - sample, axis=2)

            # Tìm BMU (Best Matching Unit)
            bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)

            # Cập nhật trọng số của BMU và các nơ-ron xung quanh BMU
            for i in range(x_size):
                for j in range(y_size):
                    # Tính khoảng cách từ nơ-ron hiện tại đến BMU
                    dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))

                    # Tính hệ số học (learning rate) và sigma (hệ số mở rộng vùng ảnh hưởng)
                    learning_factor = learning_rate * np.exp(-dist_to_bmu**2 / (2 * sigma**2))

                    # Cập nhật trọng số nơ-ron
                    weights[i, j] += learning_factor * (sample - weights[i, j])

    # Vẽ bản đồ khoảng cách
    plt.figure(figsize=(10, 10))
    plt.imshow(np.linalg.norm(weights - np.mean(weights, axis=(0, 1)), axis=2), cmap='coolwarm')
    plt.colorbar(label='Distance')
    plt.title("Self-Organizing Map (Kohonen)")
    
    # Lưu biểu đồ
    image_filename = 'kohonen_plot.png'
    image_path = f"images/{image_filename}"  # Đường dẫn tương đối từ thư mục 'static'
    plt.savefig(f"static/{image_path}")
    plt.close()

    # Trả về kết quả và đường dẫn hình ảnh
    return "Thuật toán Kohonen đã hoàn thành.", image_path
