import pandas as pd

def preprocess_data(data):
    # Xóa giá trị null
    data = data.dropna()

    # Chỉ lấy các cột số
    numeric_data = data.select_dtypes(include=['number'])

    # Tính toán ma trận tương quan cho các cột số
    correlation_matrix = numeric_data.corr()

    # Chuyển bảng mô tả và ma trận tương quan sang HTML
    describe_html = numeric_data.describe().to_html()
    correlation_html = correlation_matrix.to_html()

    # Kết hợp cả hai bảng vào một chuỗi HTML
    combined_html = (
        "<h3>Bảng mô tả dữ liệu:</h3>" +
        describe_html +
        "<h3>Ma trận tương quan:</h3>" +
        correlation_html
    )

    return combined_html  # Trả về HTML hiển thị
