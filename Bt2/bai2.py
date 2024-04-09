from re import T
import streamlit as st
import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, average_precision_score


st.title('Image Retriveval')
st.markdown("## Đinh Hoàng Tâm Bình - 21521873")

# Trích xuất đặc trưng hog từ ảnh
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    # Resize the image to reduce computational load
    image_resized = cv2.resize(image, (128, 128))
    hog_features = hog.compute(image_resized)
    return hog_features.flatten()

# Điều chỉnh kích thước của vectơ đặc trưng hog
def adjust_hog_features_size(features, target_length):
    current_length = len(features)
    if current_length < target_length:
        # If the current length is shorter than the target length, pad zeros to the end
        features = np.pad(features, (0, target_length - current_length), 'constant')
    elif current_length > target_length:
        # If the current length is longer than the target length, truncate the vector
        features = features[:target_length]
    return features

# Nhập images từ folder và lưu trữ tên cho bước tính P, R, AP
def load_images(folder_path):
    images = []
    names=[]
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                # Extract the name before the underscore
                name = filename.split('_')[0]
                names.append(name)
    return images,names


# Đoạn code chính
# Lấy ảnh từ folder query
st.write("Chọn ảnh từ Query")
uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png'], accept_multiple_files=False)
if uploaded_file is not None:
    uploaded_name = uploaded_file.name.split('_')[0] #Lưu tên
    query_image = np.array(cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1))
    img_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Ảnh truy vấn", use_column_width=True)


    features = extract_hog_features(query_image)

    target_length = 1000
    features = adjust_hog_features_size(features, target_length)

    database_path = './Bt2/database-20240322T015516Z-001/database'
    dt_images, dt_names = load_images(database_path) #Lấy ảnh + lưu tên
    st.write(f"Số lượng ảnh trong database: {len(dt_images)}")

    # Tính độ tương đồng giữa ảnh chọn và ảnh trong database
    similarities = []
    for img in dt_images:
        dt_features = extract_hog_features(img)
        dt_features = adjust_hog_features_size(dt_features, target_length)
        similarity = cosine_similarity([features], [dt_features])
        similarities.append(similarity[0][0])

    top_simi = st.selectbox("Chọn số lượng ảnh trả về", [5, 10, 15, 20], index = 1)
    top_img = np.argsort(similarities)[::-1][:top_simi]
    st.write(top_img)

    relevant_list = [0]*top_simi # Tạo list để lưu xem ảnh nào là ảnh đúng

    for tmp in top_img: #Check trong top_img, ảnh nào đúng thì gán 1 vào relevant_list tương ứng
        if dt_names[tmp] == uploaded_name:
            relevant_list[np.where(top_img == tmp)[0][0]] = 1

    sum_re = relevant_list.count(1) #Tính tổng có bn ảnh đúng
    base_re = [20, 20, 10]

    Precisions = [] #Lưu lại Precision để tính AP
    st.subheader(f"Top {top_simi} ảnh có độ tương đồng cao nhất!")

    count_relevant = 0 #Đếm số ảnh đúng đã xét
    for i in top_img:
        img_current = cv2.cvtColor(dt_images[i], cv2.COLOR_BGR2RGB)

        st.image(img_current, caption=f"{dt_names[i]}", use_column_width=True)
        cur_index = np.where(top_img == i)[0][0] #Lấy index của ảnh hiện tại
        if relevant_list[cur_index] == 1: #Nếu là ảnh đúng thì số ảnh đúng đã xét tăng 1
            count_relevant += 1
        cur_P = count_relevant/(cur_index+1) #Presicion = số ảnh đúng đã xét/số ảnh đã xét
        Precisions.append(cur_P)
        if dt_names[i] == "accordion":
            cur_re = base_re[0]
        elif dt_names[i] == "airplane":
            cur_re = base_re[1]
        else:
            cur_re = base_re[2]
        cur_R = count_relevant/cur_re #Recall = số ảnh đúng đã xét/số ảnh đúng
        st.write(f"Precision: {cur_P:.2f}, Recall: {cur_R:.2f}")

    AP = 0
    avg = 0
    for i in relevant_list:
        if i == 1:
            avg += 1
            AP += Precisions[relevant_list.index(i)]
    AP /= avg
    st.write(f"Average Precision: {AP:.2f}")
