import urllib.request
from torchvision import transforms as T

prepro_test = T.Compose([
    T.ToTensor(),
    T.Grayscale(num_output_channels=1)
])

def preprocess(img):
    return prepro_test(img)