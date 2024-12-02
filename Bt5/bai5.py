from cProfile import label
import time
from unittest import result
from cv2 import CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD
import torch
import torchvision
#from torchvision.models.detection import FasterRCNN
#from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import streamlit as st

#git clone https://github.com/ultralytics/yolov5
#cd yolov5
#pip install -r requirements.txt

st.title('Thuc Hanh 5')
st.markdown("## Đinh Hoàng Tâm Bình - 21521873")



uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png'], accept_multiple_files=False)
if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    st.image(uploaded_file, caption='Ảnh đã tải lên.', use_column_width=True)
    image = Image.open(uploaded_file)

    function = 3
    exs_y = 1

    # ----------------1-----------------
    #Thg của fasterrcnn
    st.write("FasterRCNN")
    if function == 3:

        start_time = time.time()
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
        #st.write(boxes)
        labels = preds[0]['labels']
        #st.write(labels)
        scores = preds[0]['scores']
        #st.write(scores)

        exs = 2
        image_fasterrcnn = image.copy()
        #Lính bắn tỉa
        if exs == 1:
            for box, label, score in zip(boxes, labels, scores):
                if label == 20 and score >= 0.65:
                    draw = ImageDraw.Draw(image_fasterrcnn)
                    draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=3)
                    del draw
            st.image(image, caption='Ảnh với bounding box.', use_column_width=True)
        elif exs == 2:
            count = 0
            for box, label, score in zip(boxes, labels, scores):
                if label == 1 and score >= 0.7:
                    count+=1
                    draw = ImageDraw.Draw(image_fasterrcnn)
                    draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=3)
                    del draw
            st.image(image_fasterrcnn, caption='Ảnh với bounding box.', use_column_width=True)
            st.write("Số người trong ảnh: ", count)
            st.write(f"Inference time: ", inference_time, " seconds")
        else:
            for box, label, score in zip(boxes, labels, scores):
                if score >= 0.1:
                    draw = ImageDraw.Draw(image_fasterrcnn)
                    draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=3)
                    del draw
            st.image(image, caption='Ảnh với bounding box.', use_column_width=True)


    # ----------------2-----------------
    st.write("YoloV5s")
    model_v5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    start_time_yolo = time.time()

    # conf
       
    model_v5.conf = 0.1

    results = model_v5(image, size=640)
    end_time_yolo = time.time()
    inference_time_yolo = end_time_yolo - start_time_yolo
    #Người là label 0
    pred_yolo = results.pred[0].cpu().numpy()
    boxes_yolo = pred_yolo[:, :4]
    scores_yolo = pred_yolo[:, 4]
    labels_yolo = pred_yolo[:, 5]
    # st.write(boxes_yolo)
    # st.write(scores_yolo)
    st.write(labels_yolo)

    image_yolo = image.copy()
    count_yolo = 0

    if exs_y == 1:
        for box, label, score in zip(boxes_yolo, labels_yolo, scores_yolo):
            if label == 0 and score >= 0.1:
                draw = ImageDraw.Draw(image_yolo)
                count_yolo+=1
                draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=3)
                del draw

        st.write("Số người trong ảnh: ", count_yolo, " người.")
        st.write("Inferece time: ", inference_time_yolo, " seconds.")
    else:
        for box, label, score in zip(boxes_yolo, labels_yolo, scores_yolo):
            
                draw = ImageDraw.Draw(image_yolo)
                count_yolo+=1
                draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=3)
                del draw



    
    st.image(image_yolo, caption='Ảnh với bounding box.', use_column_width=True)
    #st.image(results.render()[0], caption='Ảnh với bounding box.', use_column_width=True)
    