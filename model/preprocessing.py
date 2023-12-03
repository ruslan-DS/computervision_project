from torchvision import transforms as T
from torchvision import io

func_trnsf = T.Compose([
    T.ToTensor(),
    T.ToPILImage()
])
def make_transform(image):
    return func_trnsf(image)