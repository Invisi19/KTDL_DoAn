from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import os

def decision_tree_algorithm(data):
    # Kiểm tra nếu dữ liệu chứa giá trị phân loại và xử lý
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
            encoders[col] = encoder

    X = data.iloc[:, :-1]  # Đặc trưng
    y = data.iloc[:, -1]   # Nhãn

    # Huấn luyện mô hình Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # Xuất các quy tắc cây dưới dạng văn bản
    tree_rules = export_text(model, feature_names=list(X.columns))

    # Vẽ biểu đồ cây
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=list(X.columns), class_names=model.classes_.astype(str), filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Visualization")

    # Lưu biểu đồ vào thư mục static
    image_filename = 'decision_tree_plot.png'
    image_path = f"images/{image_filename}"  # Đường dẫn tương đối từ static

    # Đảm bảo thư mục 'static/images' tồn tại
    os.makedirs('static/images', exist_ok=True)
    plt.savefig(f"static/{image_path}")
    plt.close()

    # Trả về các quy tắc cây và đường dẫn hình ảnh
    return f"<pre>{tree_rules}</pre>", image_path

# Ví dụ cách sử dụng:
# data = pd.DataFrame({
#     'Feature1': ['A', 'B', 'A', 'C'],
#     'Feature2': ['X', 'Y', 'X', 'Z'],
#     'Label': ['Yes', 'No', 'Yes', 'No']
# })
# result = decision_tree_algorithm(data)
# print(result)
