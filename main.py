# Import các thư viện cần thiết
import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googletrans import Translator

# Hàm chuẩn hóa và tiền xử lý văn bản
def preprocess_text(text):
    """Chuyển văn bản về chữ thường và thực hiện các bước tiền xử lý khác nếu cần"""
    return text.lower()

# Hàm dịch văn bản sang tiếng Anh
def translate_to_english(text):
    """Dịch văn bản sang tiếng Anh"""
    translator = Translator()
    translated = translator.translate(text, src='auto', dest='en')  # Dịch sang tiếng Anh
    return translated.text

# Tải mô hình TensorFlow và Tokenizer đã huấn luyện
model = load_model("Model/toxic_comment_model.h5")  # Tải mô hình TensorFlow
tokenizer = joblib.load("Model/tokenizer.pkl")  # Tải Tokenizer
max_len = 100  # Độ dài tối đa cho padding

# Tiêu đề ứng dụng
st.set_page_config(page_title="Toxic Comment Classifier", page_icon="💬", layout="wide")
st.title("🌟 Toxic Comment Classifier 🌟")
st.write("## Kiểm tra các comment có phải toxic hay không!")

# Hướng dẫn sử dụng
st.markdown("""
    #### Cách sử dụng:
    1. Nhập câu comment mà bạn muốn kiểm tra vào ô bên dưới.
    2. Sau khi nhập xong, hệ thống sẽ dự đoán liệu câu comment đó có phải là toxic hay không.
    3. Kết quả sẽ được hiển thị ngay dưới ô nhập liệu.
""")

# Nhập câu comment từ người dùng
comment = st.text_input("Nhập câu comment của bạn để kiểm tra:")

# Thêm nút kiểm tra để dự đoán
if st.button("Kiểm tra") or comment:
    if comment:
        # Dịch comment sang tiếng Anh
        translated_comment = translate_to_english(comment)
        st.write(f"**Câu comment gốc**: {comment}")
        st.write(f"**Câu comment đã dịch sang tiếng Anh**: {translated_comment}")

        # Tiền xử lý comment đã dịch
        processed_comment = preprocess_text(translated_comment)

        # Chuyển đổi comment thành chuỗi số và thực hiện padding
        comment_seq = tokenizer.texts_to_sequences([processed_comment])
        comment_pad = pad_sequences(comment_seq, maxlen=max_len, padding="post", truncating="post")

        # Dự đoán với mô hình TensorFlow
        prediction_prob = model.predict(comment_pad)  # Trả về xác suất
        prediction = (prediction_prob > 0.5).astype("int32")[0][0]  # Chuyển xác suất thành nhãn (0 hoặc 1)

        # Hiển thị kết quả dự đoán với màu sắc và biểu tượng
        if prediction == 0:
            st.success("✅ Đây là comment **không toxic**.", icon="✅")
        else:
            st.error("❌ Đây là comment **toxic**.", icon="❌")

        # Hiển thị các chi tiết dự đoán
        st.write("#### Câu comment đã được xử lý như sau:")
        st.write(f"**Câu comment đã tiền xử lý**: {processed_comment}")
        st.write(f"**Xác suất dự đoán Toxic**: {prediction_prob[0][0]:.2f}")
    else:
        st.warning("Vui lòng nhập một comment để kiểm tra.")

# Thêm footer hoặc thông tin bổ sung
st.markdown("""
    ---
    ### 💡 Thông tin:
    - Mô hình này được huấn luyện sử dụng TensorFlow và Tokenizer để phân loại các comment toxic.
    - Bạn có thể thử nghiệm với các câu khác nhau để kiểm tra khả năng phân loại của mô hình.
""")
