import argparse
import os
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from datasetAugmentation import IsImageFile
from model import Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dithered dataset')
    parser.add_argument('--dithering_type', default='FloydSteinberg', type=str, help='None, Riemersma or FloydSteinberg')
    parser.add_argument('--model_name', default='epoch_FloydSteinberg_100.pt', type=str, help='model filename')
    opt = parser.parse_args()

    dithering_type = opt.dithering_type
    model_name = opt.model_name

    path = 'test images/'
    images_name = [x for x in listdir(path) if IsImageFile(x)]
    model = Net()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + model_name))

    out_path = 'test results/' + dithering_type + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for image_name in tqdm(images_name, desc='convert LR images to HR images'):

        img = Image.open(path + image_name).convert('RGB')
        image = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image)
        out = out.cpu()
        out_img = out.data[0].numpy()
        out_img *= 255.0
        out_img = out_img.clip(0, 255)
        out_img = np.uint8(out_img)
        out_img = np.swapaxes(out_img, 0, 2)
        out_img = np.swapaxes(out_img, 0, 1)
        out_img = Image.fromarray(out_img, mode='RGB')
        #out_img = out_img.convert('RGB')
        out_filename = out_path + image_name
        if os.path.isfile(out_filename):
            os.remove(out_filename)
        out_img.save(out_filename)
