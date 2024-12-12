import pandas as pd
import numpy as np

def bayes_algorithm(data, feature_list=None, feature_values=None, laplace_smoothing=False):
    # Input validation
    if not feature_list or not feature_values:
        return {
            'error': "Insufficient information for prediction",
            'prediction': None,
            'dataframe': None,
            'html_table': None
        }
    
    # Check feature list and values match
    if len(feature_list) != len(feature_values):
        return {
            'error': "Number of features and values do not match",
            'prediction': None,
            'dataframe': None,
            'html_table': None
        }
    
    # Identify target column (last column)
    target_column = data.columns[-1]
    
    # Tạo một dictionary để lưu trữ xác suất của từng lớp
    class_probabilities = {}
    
    # Lấy danh sách các lớp (giá trị của cột mục tiêu)
    classes = data[target_column].unique()
    
    # Tính xác suất ban đầu của từng lớp
    total_samples = len(data)
    initial_class_probabilities = {}
    for cls in classes:
        initial_class_probabilities[cls] = len(data[data[target_column] == cls]) / total_samples
    
    # Tính xác suất có điều kiện cho từng đặc trưng
    for cls in classes:
        # Lọc dữ liệu cho lớp hiện tại
        class_data = data[data[target_column] == cls]
        
        # Khởi tạo xác suất ban đầu của lớp
        probability = initial_class_probabilities[cls]
        
        # Tính xác suất cho từng đặc trưng
        for feature, value in zip(feature_list, feature_values):
            # Kiểm tra xem giá trị đặc trưng có tồn tại trong dữ liệu không
            if value not in data[feature].unique():
                # Nếu không tồn tại, sử dụng phân phối đều
                probability *= 1 / len(data[feature].unique())
            else:
                # Nếu tồn tại, tính xác suất có điều kiện
                feature_count = len(class_data[class_data[feature] == value])
                total_count = len(class_data)
                
                # Áp dụng Laplace smoothing nếu được yêu cầu
                if laplace_smoothing:
                    unique_feature_values = len(data[feature].unique())
                    probability *= (feature_count + 1) / (total_count + unique_feature_values)
                else:
                    probability *= feature_count / total_count
        
        # Lưu xác suất của lớp
        class_probabilities[cls] = probability
    
    # Chọn lớp có xác suất cao nhất
    predicted_class = max(class_probabilities, key=class_probabilities.get)
    
    # Chuẩn bị dữ liệu kết quả để định dạng
    result_data = [
        {
            'Đặc trưng': feature,
            'Giá trị': value
        } for feature, value in zip(feature_list, feature_values)
    ]
    
    # Thêm dự đoán vào kết quả
    result_data.append({
        'Đặc trưng': 'Kết quả dự đoán',
        'Giá trị': str(predicted_class)
    })
    
    # Tạo DataFrame
    result_df = pd.DataFrame(result_data)
    
    # Tạo HTML table
    html_table = result_df.to_html(
        classes='table table-striped table-bordered', 
        index=False, 
        escape=False
    )
    
    return html_table

# Ví dụ sử dụng
# Đọc dữ liệu
# data = pd.read_csv('bayes-data.csv')
# result = bayes_algorithm(data, 
#                          feature_list=['Outlook', 'Temperature', 'Humidity', 'Wind'], 
#                          feature_values=['Rainy', 'Mild', 'High', 'Strong'])
# print(result)