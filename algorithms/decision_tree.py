from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

def decision_tree_algorithm(data):
    X = data.iloc[:, :-1]  # Đặc trưng
    y = data.iloc[:, -1]   # Nhãn

    # Huấn luyện mô hình Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # Xuất các quy tắc cây dưới dạng văn bản
    tree_rules = export_text(model, feature_names=list(X.columns))

    # Vẽ biểu đồ cây
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=list(X.columns), class_names=model.classes_, filled=True,rounded=True,fontsize=10)
    plt.title("Decision Tree Visualization")

    # Lưu biểu đồ vào thư mục static
    image_filename = 'decision_tree_plot.png'
    image_path = f"images/{image_filename}"  # Đường dẫn tương đối từ static
    plt.savefig(f"static/{image_path}")
    plt.close()

    # Trả về các quy tắc cây và đường dẫn hình ảnh
    return f"<pre>{tree_rules}</pre>", image_path

