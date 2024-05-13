from ultralytics import YOLO
import cv2
import base64
from PIL import Image
from io import BytesIO

import json

model = YOLO('yolov8n.pt')

def draw_bbox(image):
    results = model(image)
    boxes = results[0].boxes
    class_id = boxes.cls.tolist()
    box = boxes.xyxy
    for item in range(len(class_id)):
        class_name = results[0].names[class_id[item]]
        bbox = box[item]
        bbox = [int(coord) for coord in bbox]
        
        #Draw bbox and class name
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        text = class_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        cv2.putText(image, text, (bbox[0], bbox[1] - 5), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        # print(f'Class name: {class_name}, bbox: {bbox}')
    
    return image
