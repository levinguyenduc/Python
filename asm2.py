import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu từ file CSV vào DataFrame
data = pd.read_csv('data.csv')

# Xử lý dữ liệu trùng lặp
data.drop_duplicates(inplace=True)

# # Xử lý giá trị rỗng (null)
imputer = SimpleImputer(strategy='mean')
data['column_name'] = imputer.fit_transform(data[['column_name']])

# Xử lý giá trị ngoại lai (outliers)
Q1 = data['column_name'].quantile(0.25)
Q3 = data['column_name'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['column_name'] >= Q1 - 1.5 * IQR) & (data['column_name'] <= Q3 + 1.5 * IQR)]

# Chuyển đổi kiểu dữ liệu
data['date_column'] = pd.to_datetime(data['date_column'])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data['numeric_column'] = scaler.fit_transform(data[['numeric_column']])

# Xử lý biến phân loại (categorical variables) - One-Hot Encoding
data = pd.get_dummies(data, columns=['categorical_column'])

# In ra thông tin của DataFrame sau khi xử lý
print(data.head())