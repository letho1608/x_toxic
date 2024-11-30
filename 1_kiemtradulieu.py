import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu và hiển thị thông tin cơ bản
data = pd.read_csv('Data/Data_raw.csv')
print("="*100)
print("THÔNG TIN DỮ LIỆU BAN ĐẦU")
print("="*100)
data.info()
print("-"*50)
print("5 dòng đầu tiên của dữ liệu:")
print(data.head(5))
print("-"*50)
print("5 dòng cuối cùng của dữ liệu:")
print(data.tail(5))
print("="*100)

# Hiển thị một dòng tweet bất kỳ
print("NỘI DUNG DÒNG TWEET THỨ 56600")
print("="*100)
print(data.loc[56600, 'tweet'])
print("="*100)

# Kiểm tra và xử lý dữ liệu
print("KIỂM TRA DỮ LIỆU")
print("="*100)
print(f"Số lượng giá trị bị thiếu: {data.isnull().sum().sum()}")
print(f"Số lượng bản ghi trùng lặp: {data.duplicated().sum()}")
print("-"*50)
if data.duplicated().sum() > 0:
    print("Danh sách các bản ghi trùng lặp:")
    print(data[data.duplicated()].sort_values(by='tweet'))
else:
    print("Không có bản ghi trùng lặp.")
print("="*100)

# Loại bỏ dữ liệu trùng lặp
data.drop_duplicates(keep='first', inplace=True)
print("THÔNG TIN DỮ LIỆU SAU KHI LOẠI BỎ TRÙNG LẶP")
print("="*100)
data.info()
