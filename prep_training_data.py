import argparse
import os
from os import listdir
from os.path import join
import subprocess
from PIL import Image
Image.MAX_IMAGE_PIXELS = 240000000
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import random
import datetime


def is_image_file(filename):
    correctExtension = any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.tif'])
    correctName = 'Greenshot' in filename
    return correctExtension

class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, dithering_type):
        super(DatasetFromFolder, self).__init__()
        self.n = 2 # upscale factor
        self.image_dir = dataset_dir + '/' + dithering_type + '/data_{:d}x'.format(self.n)
        self.target_dir = dataset_dir + '/' + dithering_type + '/target'
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.input_transform = transforms.ToTensor()
        self.target_transform = transforms.ToTensor()
        self.image_patches = []
        self.target_patches = []
        self.patch_count = 0
        # Load fixed overlay shapes
        self.shape_library = {}
        for shape_color in ['black','white']:
            for shape_type in ['transparent','grid']:
                if shape_type == 'grid':
                    scale_string = '{:d}x'.format(self.n)
                else:
                    scale_string = '1x'
                shape_filename = 'blob_{}_{}_{}.png'.format(shape_color, shape_type, scale_string)
                shape = Image.open(shape_filename).convert('RGBA')
                self.shape_library[shape_color + shape_type] = shape

    def __getitem__(self, index):
        imageFilename = self.image_filenames[index]
        targetFilename = self.target_filenames[index]
        image = Image.open(imageFilename).convert('RGB')
        target = Image.open(targetFilename).convert('RGB')
        # Extract a background plane (try to stay above 64x64, but smaller dimensions are faster)
        n = self.n
        l = 200 # length of a training square
        h,w = (int(l/n),int(l/n))
        x,y = image.size
        j = random.randint(0,(x-h)-1)
        i = random.randint(0,(y-w)-1)
        image_bg = transforms.functional.crop(image,i,j,h,w)
        target_bg = transforms.functional.crop(target,i*n,j*n,h*n,w*n)
        # Insert a random small patch from somewhere else to force a hard border
        if self.patch_count < 50:
            hp,wp = (random.randint(1,40),random.randint(1,40))
            j = random.randint(0,(x-hp)-1)
            i = random.randint(0,(y-wp)-1)
            image_patch = transforms.functional.crop(image,i,j,hp,wp)
            target_patch = transforms.functional.crop(target,i*n,j*n,hp*n,wp*n)
            self.image_patches.append(image_patch)
            self.target_patches.append(target_patch)
            self.patch_count += 1
        else:
            patchID = random.randint(0,self.patch_count-1)
            image_patch = self.image_patches[patchID]
            target_patch = self.target_patches[patchID]
            hp, wp = image_patch.size
        j = random.randint(0,(h-hp)-1)
        i = random.randint(0,(w-wp)-1)
        image_bg.paste(image_patch, (i,j))
        target_bg.paste(target_patch, (i*n,j*n))
        # Insert a white or black semi-transparent shape
        if random.randint(1,2) == 1:
            shape_color = random.choice(['black','white'])
            j = random.randint(0,(h-int(46/n))-1)
            i = random.randint(0,(w-int(46/n))-1)
            shape_grid = self.shape_library[shape_color + 'grid']
            shape_transparent = self.shape_library[shape_color + 'transparent']
            image_bg.paste(shape_grid, (i,j), shape_grid)
            target_bg.paste(shape_transparent, (i*n,j*n), shape_transparent)
        # Write out a few training examples
        if random.randint(1,1000) == 542:
            sample_filename = 'training samples/{:04d}.png'.format(index)
            if not os.path.isfile(sample_filename):
                sample = Image.new('RGB',(h*2*n+1,w*n))
                if n > 1:
                    image = image_bg.resize((h*n,w*n))
                else:
                    image = image_bg
                sample.paste(image, (0,0))
                sample.paste(target_bg, (h*n+1,0))
                sample.save(sample_filename)
        # Convert to Tensor
        image_tensor = self.input_transform(image_bg.convert('YCbCr'))
        target_tensor = self.target_transform(target_bg.convert('YCbCr'))
        return image_tensor, target_tensor

    def __len__(self):
        return len(self.image_filenames)

def generate_dataset(data_type, dithering_type):
    path = 'raw training data/' + data_type + '/'
    image_filenames = [x for x in listdir(path) if is_image_file(x)]
    minimum_size = 200 / 0.7
    upscale_factor = 2

    root = 'data/' + data_type
    path = root + '/' + dithering_type
    reference_path = path + '/target'
    if not os.path.exists(reference_path):
        os.makedirs(reference_path)
    transformed_path = path + '/data_{:d}x'.format(upscale_factor)
    if not os.path.exists(transformed_path):
        os.makedirs(transformed_path)

    input_formats = ['.jpg','.jpeg','.JPG','.JPEG','.tif']
    for image_name in tqdm(image_filenames, desc='Converting raw pictures into cropped and dithered dataset ' + data_type + '...'):
        input_filename = 'raw training data/' + data_type + '/' + image_name
        image = Image.open(input_filename)
        width, height = image.size
        if width >= minimum_size and height >= minimum_size:
            base, ext = os.path.splitext(input_filename)
            output_name = image_name.replace(ext,'.png')
            transformed_filename = transformed_path + '/' + output_name
            reference_filename = reference_path + '/' + output_name
            transform_call = ['magick','convert',input_filename,'-scale','35%','-dither',dithering_type,'-remap','PerOxyd.bmp',transformed_filename]
            reference_call = ['magick','convert',input_filename,'-scale','70%',                                                reference_filename]
            if not os.path.isfile(transformed_filename):
                subprocess.run(transform_call)
            if not os.path.isfile(reference_filename):
                subprocess.run(reference_call)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dithered dataset')
    parser.add_argument('--dithering_type', default='FloydSteinberg', type=str, help='None, Riemersma or FloydSteinberg')
    opt = parser.parse_args()
    dithering_type = opt.dithering_type

    generate_dataset(data_type='train', dithering_type=dithering_type)
    generate_dataset(data_type='val', dithering_type=dithering_type)
