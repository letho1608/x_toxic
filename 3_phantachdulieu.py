# Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split

# Tải file dữ liệu đã tiền xử lý
print("\nTải dữ liệu đã tiền xử lý từ file thư mục Data")
data_cleaned = pd.read_csv('Data/Data_cleaned.csv')
print("="*100)

# Xem trước 10 dòng đầu tiên của dữ liệu
print("\nXem trước 10 dòng đầu tiên của dữ liệu:")
print(data_cleaned.head(10))
print("="*100)

# Biến độc lập X (cột tweet_clean chứa dữ liệu đã tiền xử lý)
X = data_cleaned.loc[:, 'tweet_clean']
print("\nBiến độc lập (X) chứa dữ liệu tweet đã xử lý:")
print(X.head(10))
print("="*100)

# Biến phụ thuộc y (cột class là nhãn)
y = data_cleaned.loc[:, 'class']
print("\nBiến phụ thuộc (y) chứa nhãn:")
print(y.head(10))
print("="*100)

# Tách tập dữ liệu thành Train - Test với tỷ lệ 80% Train, 20% Test
print("\nTách tập dữ liệu thành tập Train và Test (tỷ lệ 80-20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)

# In ra kích thước của tập dữ liệu ban đầu và sau khi tách
print("\nKích thước của các tập dữ liệu:")
print(f"1. Tập ban đầu: {data_cleaned.shape}")
print("="*100)
print(f"2. Tập Train: {X_train.shape}, {y_train.shape}")
print("="*100)
print(f"3. Tập Test: {X_test.shape}, {y_test.shape}")
print("="*100)

# Lưu tập Train và Test vào các file CSV
print("\nLưu các tập Train và Test vào file CSV...")
print("="*100)
data_train = pd.concat([X_train, y_train], axis=1)
data_train.to_csv('Data/Data_train.csv', index=False)

data_test = pd.concat([X_test, y_test], axis=1)
data_test.to_csv('Data/Data_test.csv', index=False)

print("\nĐã lưu dữ liệu đã được phân vào thư mục Data")
print("="*100)
