# Import các thư viện cần thiết
import nltk
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import os

# Tải xuống tài nguyên NLTK cần thiết
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Khởi tạo các đối tượng và danh sách từ
wn = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = ['rt', 'ht', 'fb', 'amp', 'gt']  # Stopwords tùy chỉnh

# Hàm xử lý từng bước
def decontracted(text):
    """Chuyển đổi các dạng viết tắt thành dạng đầy đủ."""
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

def clear_link(text):
    """Loại bỏ các đường link và địa chỉ email."""
    text = re.sub(r'\S+@\S+', '', text)  # Loại bỏ email
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Loại bỏ link
    return text

def clear_punctuation(text):
    """Loại bỏ dấu câu."""
    return re.sub(r'[^\w\s]', '', text)

def clear_special(text):
    """Loại bỏ ký tự không phải chữ cái."""
    return re.sub(r'[^a-zA-Z\s]', ' ', text)

def clear_stopwords(text):
    """Loại bỏ stopwords."""
    words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

def black_txt(word):
    """Kiểm tra và lọc từ không cần thiết."""
    return word.lower() not in stop_words and word not in string.punctuation and word.lower() not in custom_stopwords

def fun_stemlem(text):
    """Lemmatize từng từ và loại bỏ các từ không cần thiết."""
    clean_words = [wn.lemmatize(word.lower(), pos="v") for word in text.split() if black_txt(word)]
    return ' '.join(clean_words)

def clear_noise(text):
    """Tổng hợp các bước làm sạch."""
    text = text.lower()
    text = decontracted(text)
    text = clear_link(text)
    text = clear_punctuation(text)
    text = clear_special(text)
    return text

# Hàm tổng hợp toàn bộ tiền xử lý
def prepare_data(text):
    """Chuẩn bị dữ liệu qua tất cả các bước tiền xử lý."""
    text = clear_noise(text)
    text = clear_stopwords(text)
    text = fun_stemlem(text)
    return text

# Nạp dữ liệu từ file CSV
print("\nNạp dữ liệu từ thư mục Data:")
data = pd.read_csv('Data/Data_raw.csv')
print("="*100)

# Kiểm tra dữ liệu
print("\nThông tin tập dữ liệu:")
print(data.info())
print("="*100)

# Tiền xử lý dữ liệu
print("\nBắt đầu tiền xử lý dữ liệu...")
data['tweet_clean'] = data['tweet'].apply(prepare_data)
print("="*100)

# Thống kê dữ liệu
print("\nXem trước dữ liệu đã tiền xử lý:")
print(data.head())
print("="*100)

# Kiểm tra số lượng dữ liệu rỗng
empty_clean = data['tweet_clean'].str.strip().eq('').sum()
print(f"\nSố bản ghi rỗng sau khi xử lý: {empty_clean}")
print("="*100)

# Lọc dữ liệu hợp lệ
data_cleaned = data[data['tweet_clean'].str.strip() != '']

# Kiểm tra mức độ cân bằng của tập dữ liệu
print("\nThống kê số lượng comment theo lớp:")
class_counts = data_cleaned['class'].value_counts()
print(class_counts)
print("="*100)

# Tạo thư mục Visual nếu chưa có
if not os.path.exists('Visual'):
    os.makedirs('Visual')

# Trực quan hóa
labels = ['Non-Toxic (0)', 'Toxic (1)']
plt.figure(figsize=(12, 6))
plt.suptitle("Phân phối các lớp dữ liệu", fontsize=16)

# Vẽ biểu đồ tròn
plt.subplot(1, 2, 1)
plt.pie(class_counts, labels=labels, autopct='%1.1f%%', colors=['#1E77B4', '#FF7F0E'])
plt.title("Biểu đồ phân phối lớp (Tròn)")

# Vẽ biểu đồ cột
plt.subplot(1, 2, 2)
plt.bar(labels, class_counts, color=['#1E77B4', '#FF7F0E'])
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
plt.title("Biểu đồ phân phối lớp (Cột)")

# Lưu ảnh vào thư mục Visual
plt.savefig('Visual/tienxuly_visual.png')
print("\nBiểu đồ phân phối lớp đã được lưu vào thư mục 'Visual'.")
print("="*100)

# Xuất kết quả ra file CSV
data_cleaned.to_csv('Data/Data_cleaned.csv', index=False, header=True)
print("\nDữ liệu đã được lưu vào Data/Data_cleaned.csv.")
