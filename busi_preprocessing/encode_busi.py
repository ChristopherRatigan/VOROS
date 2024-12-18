import argparse
import os
import re
import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.io import read_image

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def init_encoder():
    # Load pretrained weights
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    # Load ViT with pretrained weights
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights(weights))
    # Remove classification head
    model.heads = Identity()
    model.eval()
    return model

def encode_image(model, path, grayscale=True, image_size=(224, 224)):
    model_device = next(model.parameters()).device
    device = torch.device(model_device)
    # Default from https://pytorch.org/vision/main/models/vision_transformer.html
    mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
    std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
    # Read image
    image = read_image(path).float()
    # Make sure image dimensions are (1, w, h)
    if grayscale and len(image.shape) == 2: image = image[None, :, :]
    # For grayscale images read with RGB values all dimension will be the same
    if grayscale and len(image.shape) == 3 and not image.shape[0] == 1: image = image[0:1, :, :]
    # Center-crop the images using a window size equal to the length of the shorter edge and rescale them to (1, 224, 224)
    print(image.shape)
    center_crop = torchvision.transforms.CenterCrop(min(image.shape[-1], image.shape[-2]))
    image = center_crop(image)
    image = torchvision.transforms.functional.resize(image, size=image_size)
    # Expand image
    if grayscale: image = image.expand(3, image_size[0], image_size[1])
    # Normalize image
    image = ((image/255)-mean)/std
    # Encode image
    with torch.no_grad():
        if device.type == 'cuda': image = image.to(device)
        encoded_image = model(image[None, :, :, :])
        if device.type == 'cuda': encoded_image = encoded_image.cpu()
    return encoded_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_BUSI.py')
    parser.add_argument('--dataset_path', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_encoder()
    model.to(device)
    print(f"{args.dataset_path}")
    for root, dirs, files in os.walk(args.dataset_path):
        print("are you here")
        for file in files:
            print("Or here?")
            if re.search('\([0-9]+\).png$', file):
                label = re.split('\/', root)[-1]
                assert label in ['benign', 'malignant', 'normal'], 'Unexpected label: {}'.format(label)
                path = os.path.join(root, file)
                encoded_image = encode_image(model, path, grayscale=False)
                new_path='{}/encoded_BUSI/{}/'.format(os.path.dirname(args.dataset_path), label)
                print("hello")
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                encoded_path = new_path+re.split('\.', file)[0]+".pt"
                torch.save(encoded_image, encoded_path)
    