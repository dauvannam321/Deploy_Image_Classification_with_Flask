import cv2
import torch

def imgToTensor(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float')
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)
    return image

def predict(model, image):
    result = model(image)
    if result[0][0].item() > result[0][1].item():
        return "dog"
    else:
        return "cat"