import numpy as np
from PIL import Image
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
   
    image = image.resize((256, 256))
    # crop expects 4 coordinates representing a square box:
    image = image.crop((16, 16, 240, 240))
    image = np.array(image)/255
    
    # Normalizing images
    mean = np.array([0.485, 0.456, 0.406])
    SD = np.array([0.229, 0.224, 0.225])
    image = (image-mean)/SD

    # Reorder dimensions for pytorch
    image = np.transpose(image, (2, 0, 1))

    return image



def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = Image.open(image_path)
    image = process_image(image)
    
    # Convert 2D image into tensor
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).float()

    # Perform prediction
    model.eval()
    model.to(device)
    image.to(device)
    output = model.forward(image)
    # output is log_softmax --> take exp.
    probabilities, index = torch.exp(output).topk(topk)
    index = index.numpy()[0]
    probabilities = probabilities.detach().numpy()[0]
    
    #  Obtaining classes from index
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes = []
    for label in index:
        classes.append(idx_to_class[label])

    return probabilities, classes