from cProfile import label
import time
import torch
import torchvision
#from torchvision.models.detection import FasterRCNN
#from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import streamlit as st

st.title('Thuc Hanh 4')
st.markdown("## Đinh Hoàng Tâm Bình - 21521873")



uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png'], accept_multiple_files=False)
if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    st.image(uploaded_file, caption='Ảnh đã tải lên.', use_column_width=True)
    start_time = time.time()
    image = Image.open(uploaded_file)
    transform = transforms.Compose([transforms.ToTensor()])

    # Apply the transformations to the input image
    test_img_tensor = transform(image)
    test_img_tensor = test_img_tensor.unsqueeze(dim = 0)
    
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    preds = model(test_img_tensor)
    #st.write(preds)
    end_time = time.time()

    # Calculate the inference time
    inference_time = end_time - start_time

    boxes = preds[0]['boxes']
    st.write(boxes)
    labels = preds[0]['labels']
    #st.write(labels)
    scores = preds[0]['scores']
    st.write(scores)

    exs = 1

    #Lính bắn tỉa
    if exs == 1:
        for box, label, score in zip(boxes, labels, scores):
            if label == 20 and score >= 0.65:
                draw = ImageDraw.Draw(image)
                draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=3)
                del draw
        st.image(image, caption='Ảnh với bounding box.', use_column_width=True)
    elif exs == 2:
        count = 0
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score >= 0.7:
                count+=1
                draw = ImageDraw.Draw(image)
                draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=3)
                del draw
        st.image(image, caption='Ảnh với bounding box.', use_column_width=True)
        st.write("Số người trong ảnh: ", count)
        st.write(f"Inference time: {inference_time:.2f} seconds")
    else:
        for box, label, score in zip(boxes, labels, scores):
            if score >= 0.1:
                draw = ImageDraw.Draw(image)
                draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=3)
                del draw
        st.image(image, caption='Ảnh với bounding box.', use_column_width=True)
