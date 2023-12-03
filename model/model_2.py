from ultralytics import YOLO
import cv2
from model.preprocessing import make_transform

yolo8_model = YOLO('weights/yolo8n_186epochs.pt')

def predict_2(img):
    img = make_transform(img)
    detection = yolo8_model(img)
    detect_img = detection[0].plot()

    return detect_img