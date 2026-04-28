import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 1. Cấu hình trang
st.set_page_config(page_title="AI Model Web App", layout="wide")
st.title("Ứng dụng Phân loại AI")

# 2. Hàm tải model (Sử dụng cache để không tải lại mỗi khi nhấn nút)
@st.cache_resource
def load_my_model():
    # Đảm bảo file 'my_model.h5' nằm cùng thư mục với app.py
    return tf.keras.models.load_model('my_model.h5')

try:
    model = load_my_model()
    st.success("Model đã được tải thành công!")
except Exception as e:
    st.error(f"Không tìm thấy file model: {e}")

# 3. Upload file dữ liệu
uploaded_file = st.file_uploader("Tải lên file dữ liệu kiểm tra (CSV)", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu
    data = pd.read_csv(uploaded_file)
    st.write("Dữ liệu đã tải lên:", data.head())

    # Giả sử bạn có X_test và y_test trong file CSV
    # Bạn cần tiền xử lý dữ liệu ở đây cho khớp với lúc training
    # X = data.drop(columns=['label']) 
    # y_true = data['label']

    if st.button("Chạy dự đoán"):
        # 4. Dự đoán
        # y_pred_probs = model.predict(X)
        # y_pred = np.argmax(y_pred_probs, axis=1)
        
        st.write("Đang dự đoán...")
        # (Chèn logic dự đoán của bạn vào đây)
