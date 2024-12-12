import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kmeans_algorithm(data, n_clusters=3, max_iter=100, tolerance=1e-4):
    # Kiểm tra điều kiện dữ liệu đầu vào
    if n_clusters <= 0:
        raise ValueError("Số cụm (n_clusters) phải lớn hơn 0.")
    if n_clusters > len(data):
        raise ValueError("Số cụm (n_clusters) không được vượt quá số lượng mẫu trong dữ liệu.")
    
    # Chuyển đổi dữ liệu thành mảng numpy
    data = data.values
    n_samples, n_features = data.shape
    
    # Khởi tạo centroid ngẫu nhiên
    np.random.seed(42)
    centroids = data[np.random.choice(n_samples, n_clusters, replace=False)]
    
    for iteration in range(max_iter):
        # Tính khoảng cách giữa mỗi điểm và mỗi centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # Gán mỗi điểm vào cụm gần nhất
        labels = np.argmin(distances, axis=1)
        
        # Lưu lại centroid trước đó để kiểm tra hội tụ
        old_centroids = centroids.copy()
        
        # Cập nhật centroid mới
        for i in range(n_clusters):
            points_in_cluster = data[labels == i]
            if len(points_in_cluster) > 0:
                centroids[i] = points_in_cluster.mean(axis=0)
        
        # Kiểm tra hội tụ
        centroid_shift = np.linalg.norm(centroids - old_centroids, axis=None)
        if centroid_shift < tolerance:
            print(f"K-Means hội tụ sau {iteration + 1} vòng lặp.")
            break
    else:
        print("K-Means không hội tụ sau số vòng lặp tối đa.")
        
    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title('K-Means Clustering (Custom Implementation)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    # Lưu biểu đồ
    image_filename = 'kmeans_custom_plot.png'
    image_path = f"images/{image_filename}"  # Đường dẫn tương đối từ thư mục 'static'
    plt.savefig(f"static/{image_path}")
    plt.close()

    # Chuyển đổi kết quả thành DataFrame
    result_data = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(n_features)])
    result_data['Cluster'] = labels

    return result_data.to_html(), image_path