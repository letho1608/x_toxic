# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# Đọc tập dữ liệu huấn luyện và kiểm tra
training = "Data/Data_train.csv"
testing = "Data/Data_test.csv"

# Đọc dữ liệu từ các file CSV
train_data = pd.read_csv(training)
test_data = pd.read_csv(testing)

# Biến độc lập (X) chứa tweet_clean (dữ liệu đã xử lý)
X_train = train_data['tweet_clean']
X_test = test_data['tweet_clean']

# Biến phụ thuộc (y) chứa nhãn (class)
y_train = train_data['class']
y_test = test_data['class']

# Tokenizer: Chuyển văn bản thành các chỉ số
max_words = 5000  # Số từ tối đa
max_len = 100     # Độ dài tối đa của mỗi câu
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Chuyển văn bản thành chuỗi số và padding
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

# Chuyển nhãn sang mảng numpy
y_train = np.array(y_train)
y_test = np.array(y_test)

# Xây dựng mô hình TensorFlow
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Phân loại nhị phân (toxic hoặc non-toxic)
])

# Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Hiển thị kiến trúc mô hình
model.summary()

# Huấn luyện mô hình
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,  # Sử dụng 20% dữ liệu train làm validation
    epochs=10,             # Số epochs
    batch_size=32,
    verbose=1
)

# Tạo thư mục lưu hình ảnh nếu chưa tồn tại
if not os.path.exists('Visual'):
    os.makedirs('Visual')

# Biểu đồ Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Biểu đồ Loss qua các Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Visual/loss_curve.png")
plt.close()

# Biểu đồ Accuracy Curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title("Biểu đồ Accuracy qua các Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Visual/accuracy_curve.png")
plt.close()

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=1)
print(f"Độ chính xác trên tập kiểm tra: {test_acc * 100:.2f}%")

# Dự đoán trên tập kiểm tra
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

# In báo cáo phân loại
print("Báo cáo phân loại:")
report = classification_report(y_test, y_pred, target_names=["Non-Toxic", "Toxic"])
print(report)

# Lưu báo cáo phân loại vào file
with open("Visual/classification_report.txt", "w") as f:
    f.write(report)

# Hiển thị Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Toxic", "Toxic"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("Visual/huanluyen_visual_confusion_matrix.png")
plt.close()

# Lưu mô hình và Tokenizer
if not os.path.exists('Model'):
    os.makedirs('Model')

model.save("Model/toxic_comment_model.h5")
joblib.dump(tokenizer, "Model/tokenizer.pkl")

print("=" * 100)
print("Đã lưu mô hình, Tokenizer và các biểu đồ vào thư mục Model và Visual")
print("=" * 100)
