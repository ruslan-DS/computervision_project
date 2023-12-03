import torch
from model.preprocessing import make_transform

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/bestweights_yolov5l.pt', force_reload=True)
# model = torch.hub.load('yolov5', 'custom', path='weights/bestweights_yolov5l.pt', source='local')

def predict_1(image):

    image = make_transform(image)


    model.eval()
    predict = model(image)

    return predict
