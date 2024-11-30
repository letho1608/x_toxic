# Import thư viện cần thiết
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hàm chuẩn hóa và tiền xử lý văn bản
def preprocess_text(text):
    """Chuyển văn bản về chữ thường và thực hiện các bước tiền xử lý khác nếu cần"""
    return text.lower()

# Tải mô hình TensorFlow và Tokenizer đã huấn luyện
model = load_model("Model/toxic_comment_model.h5")  # Tải mô hình TensorFlow
tokenizer = joblib.load("Model/tokenizer.pkl")  # Tải Tokenizer

# Nhập comment từ người dùng
print("Nhập vào câu comment để kiểm tra:")
comment = input("Nhập câu comment: ").strip()  # Lấy input từ người dùng và loại bỏ khoảng trắng thừa

# Tiền xử lý comment
processed_comment = preprocess_text(comment)

# Chuyển đổi comment thành chuỗi số và thực hiện padding
max_len = 100  # Đảm bảo độ dài chuỗi phù hợp với mô hình
comment_seq = tokenizer.texts_to_sequences([processed_comment])  # Chuyển văn bản sang chuỗi số
comment_pad = pad_sequences(comment_seq, maxlen=max_len, padding="post", truncating="post")  # Padding

# Dự đoán với mô hình TensorFlow
prediction_prob = model.predict(comment_pad)  # Trả về xác suất
prediction = (prediction_prob > 0.5).astype("int32")[0][0]  # Chuyển xác suất thành nhãn (0 hoặc 1)

# In kết quả dự đoán
print("="*100)
print(f"Câu comment: {comment}")
print(f"Chuỗi sau xử lý: {processed_comment}")
print(f"Chuỗi số sau Tokenizer: {comment_seq}")
print(f"Vector sau padding: {comment_pad}")
print("="*100)

if prediction == 0:
    print("Đây là comment không toxic.")
else:
    print("Đây là comment toxic.")
print("="*100)
