import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Cấu hình giao diện
st.set_page_config(page_title="Animal Classification AI", layout="centered")
st.title("🐾 Ứng dụng Phân loại Động vật")
st.write("Tải lên một hình ảnh để AI dự đoán loài vật đó là gì.")

# 2. Định nghĩa danh sách nhãn (Bạn hãy thay đổi thứ tự này cho đúng với dữ liệu của bạn)
# Ví dụ: ['Cat', 'Dog', 'Leopard', 'Lion', 'Tiger']
CLASS_NAMES = ['Cat', 'Dog', 'Leopard', 'Lion', 'Tiger'] 

# 3. Hàm tải Model
@st.cache_resource
def load_model():
    # Đảm bảo bạn đã tải file 'my_model.h5' lên cùng thư mục
    return tf.keras.models.load_model('my_model.h5')

try:
    model = load_model()
    st.sidebar.success("✅ Model đã sẵn sàng!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi tải model: {e}")
    st.stop()

# 4. Upload file
uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh đã chọn
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    if st.button("Dự đoán ngay"):
        with st.spinner('Đang phân tích...'):
            # --- TIỀN XỬ LÝ ẢNH (Giống hệt trong file .ipynb) ---
            # 1. Resize về 250x250
            img_resized = image.resize((250, 250))
            # 2. Chuyển sang mảng numpy và đảm bảo có 3 kênh màu RGB
            img_array = np.array(img_resized.convert('RGB'))
            # 3. Chuẩn hóa (Normalizing) chia cho 255.0
            img_array = img_array.astype('float32') / 255.0
            # 4. Thêm chiều batch (1, 250, 250, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # --- DỰ ĐOÁN ---
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) # Lấy xác suất
            class_idx = np.argmax(predictions[0]) # Lấy chỉ số lớp cao nhất
            
            # --- HIỂN THỊ KẾT QUẢ ---
            st.success(f"Kết quả: **{CLASS_NAMES[class_idx]}**")
            st.write(f"Độ tin cậy: {np.max(predictions[0]) * 100:.2f}%")
            
            # Hiển thị biểu đồ xác suất cho các lớp khác
            st.bar_chart(pd.DataFrame(predictions[0], index=CLASS_NAMES, columns=["Độ tin cậy"]))
