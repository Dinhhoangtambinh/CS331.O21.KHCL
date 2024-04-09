# Kích thước của con chó đầu tiên 332 x 382
# 640 x 638 -> 224 x 224
# 116 x 134

from audioop import mul
from math import e
from turtle import width
from uu import decode
import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw


st.title('Thuc Hanh 3')
st.markdown("## Đinh Hoàng Tâm Bình - 21521873")

model = MobileNetV2(weights='imagenet')

exs = st.sidebar.selectbox(
    'Bai Tap',
    ('Cau 1', 'Cau 2')
)

def Crop_image(image, w, h, step_w, step_h, exs):
    i = 0
    j = 0
    height, width = image.size
    cat_names = ["lynx"]
    bbs_cat = []
    dog_names = ["golden_retriever","Great_Pyrenees", "Chihuahua", "Pembroke"]
    bbs_dog = []
    while i< height - h:
        
        #st.write("i: ", i)
        while j < width - w:

            #st.write("j: ", j)
            #crop_img = image[i:i+h, j:j+w] #crop the image
            crop_img = image.crop((j, i, j+w, i+h))
            crop_img_arr = np.array(crop_img.resize((224, 224))) / 255.0
            crop_img_arr = np.expand_dims(crop_img_arr, axis=0)
            #st.image(crop_img, use_column_width=True) #Xuat ra
    
            # Make prediction
            predictions = model.predict(crop_img_arr)
            decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=1)[0]
            # st.image(crop_img, caption='Ảnh đã crop.', use_column_width=True)
            # st.write(decoded_predictions[0][1])

            if exs == "Cau 1":
                if decoded_predictions[0][1] in dog_names:
                    st.write("Co con cho: " + decoded_predictions[0][1])
                    tmp = image.copy()
                    draw = ImageDraw.Draw(tmp)
                    draw.rectangle([j, i, j + w, i + h], outline="red", width=3)
                    del draw
                    st.image(tmp, caption='Ảnh với bounding box cho đối tượng Dog.', use_column_width=True)

            else:
                if decoded_predictions[0][1] in cat_names:
                    bbs_cat.append([j, i, j + w, i + h])
                elif decoded_predictions[0][1] in dog_names:
                    bbs_dog.append([j, i, j + w, i + h])
            j += step_w
        j = 0
        i += step_h

    if exs == "Cau 2":
        tmp = image.copy()
        draw = ImageDraw.Draw(tmp)
        for bb in bbs_cat:
            draw.rectangle(bb, outline="green", width=3)
        for bb in bbs_dog:
            draw.rectangle(bb, outline="red", width=3)
        del draw
        st.image(tmp, caption='Ảnh với bounding box cho đối tượng Cat và Dog.', use_column_width=True)

uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png'], accept_multiple_files=False)

if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    st.image(uploaded_file, caption='Ảnh đã tải lên.', use_column_width=True)
    image = Image.open(uploaded_file)
    
    if exs == 'Cau 1':
        #size cho cau 1: 500 - 500
        Crop_image(image, 350, 400, 50, 50, exs)
        #Crop_image(image, 80, 120, 30, 30, exs)

    else:
        #size cho cau 2: 
        Crop_image(image, 75, 120, 80, 80, exs)