import argparse
import os
from os import listdir
from os.path import join
import subprocess
from PIL import Image, ImageCms
import numpy as np
import random
import datetime
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 240000000

def IsImageFile(filename):
    """
    Check if a file has a valid image extension.
    """
    validExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    base, ext = os.path.splitext(filename)
    return ext.lower() in validExtensions

def RgbToLab(pilImage):
    """
    Convert a PIL RGB image to LAB using Pillow's ImageCms.
    Returns a NumPy array of shape (H, W, 3).
    """
    srgbProfile = ImageCms.createProfile("sRGB")
    labProfile = ImageCms.createProfile("LAB")
    transform = ImageCms.buildTransformFromOpenProfiles(
        srgbProfile, labProfile, "RGB", "LAB"
    )
    labImage = ImageCms.applyTransform(pilImage, transform)
    return np.array(labImage)

def MeasureDistanceBetweenImageAndPalette(inputFilePath, paletteFilePath='PerOxyd.bmp'):
    """
    Loads the dithering palette from 'PerOxyd.bmp', converts both the palette
    and the input image to LAB, computes the average L2 distance to the closest
    palette color. Returns the average distance (float).
    """
    # 1) Load the palette and get unique colors
    paletteImage = Image.open(paletteFilePath).convert("RGB")
    paletteColors = list(set(paletteImage.getdata()))  # Unique (R,G,B) tuples
    paletteSmall = Image.new("RGB", (len(paletteColors), 1))
    paletteSmall.putdata(paletteColors)
    paletteLab = RgbToLab(paletteSmall)[0, :, :]  # shape: (lenColors, 3)

    # 2) Convert the input image to LAB
    inputImage = Image.open(inputFilePath).convert("RGB")
    h = inputImage.height // 8
    w = inputImage.width // 8
    downsampledImage = inputImage.resize((w, h), resample=0)
    inputLab = RgbToLab(downsampledImage)  # shape: (H, W, 3)
    height, width, _ = inputLab.shape
    inputLab2D = inputLab.reshape(-1, 3)  # shape: (H*W, 3)

    # 3) For each pixel, find nearest palette color in LAB
    paletteLabExpanded = paletteLab[:, np.newaxis, :]
    inputLabExpanded = inputLab2D[np.newaxis, :, :]
    diff = paletteLabExpanded - inputLabExpanded
    distSq = np.sum(diff**2, axis=2)
    minDistSq = np.min(distSq, axis=0)
    minDist = np.sqrt(minDistSq)

    avgDistance = float(np.mean(minDist))
    return avgDistance

def find_feasible_scales(width, height, upScaleFactor, minimumSize, num_attempts=9):
    """
    For an image of size (width, height), compute a list of all feasible
    random scales (one for each 'i in range(num_attempts)'), such that
    the downsampled dimensions remain >= minimumSize. Each scale is chosen
    randomly in [i/10, (i+1)/10).

    Returns:
      A list of (i, random_scale, transform_width, transform_height).
      If empty, there's no feasible scale.
    """
    feasible_list = []
    for i in range(num_attempts):
        random_scale = random.uniform(i / 10, (i + 1) / 10)
        transform_scale = random_scale / upScaleFactor
        transform_height = int(transform_scale * height)
        transform_width  = int(transform_scale * width)

        if transform_height >= minimumSize and transform_width >= minimumSize:
            # We keep this scale
            feasible_list.append((i, random_scale, transform_width, transform_height))

    return feasible_list

def dither_and_save(
    inputFilePath,
    baseFilename,
    ditheringType,
    upScaleFactor,
    i,
    transform_width,
    transform_height,
    dataType,
    targetPathTemplate,
    existingImages
):
    """
    Performs the actual ImageMagick calls to produce the
    low-resolution (dithered) and high-resolution (reference) images.
    This version now checks a dictionary/set (existingImages) to see
    if we already generated them.
    """
    # Recompute the target folder paths based on the chosen dataType
    targetPath = targetPathTemplate.format(dataType)
    referencePath = targetPath + '/target'
    transformedPath = targetPath + f'/data_{upScaleFactor}x'

    os.makedirs(referencePath, exist_ok=True)
    os.makedirs(transformedPath, exist_ok=True)

    # Figure out the exact output sizes
    reference_height = transform_height * upScaleFactor
    reference_width = transform_width * upScaleFactor

    # Filenames
    transformedName = f'{baseFilename}-{i}.png'
    referenceName = f'{baseFilename}-{i}.png'
    transformedFilePath = f'{transformedPath}/{transformedName}'
    referenceFilePath = f'{referencePath}/{referenceName}'

    # Check via the dictionary/set if we've already processed them
    # Key could be exactly the filename, or the path. 
    # Here, we'll just store filenames as keys.
    if transformedName in existingImages or referenceName in existingImages:
        #print(f'...this iteration ({transformedName}) already exists. Skipped.')
        return

    # ImageMagick calls
    transformText = f'{transform_width}x{transform_height}!'
    referenceText = f'{reference_width}x{reference_height}!'

    transformCall = [
        'magick',
        inputFilePath,
        '-scale', transformText,
        '-dither', ditheringType,
        '-remap', 'PerOxyd.bmp',
        transformedFilePath
    ]
    referenceCall = [
        'magick',
        inputFilePath,
        '-scale', referenceText,
        referenceFilePath
    ]

    print(f'...producing image pair {baseFilename}-{i}.png...')
    # Actually run them
    subprocess.run(transformCall)
    subprocess.run(referenceCall)
    print('...complete.')

def build_existing_image_dict(targetPathTemplate, dataTypes):
    """
    Recursively parse images in the train and val folders (given
    by 'dataTypes'), collecting all filenames in a set. This can
    be used later to skip generating existing images.
    """
    existing_images = set()

    for dataType in dataTypes:
        # For each data type, get the top directory
        topFolder = targetPathTemplate.format(dataType)
        # Recursively walk
        for root, dirs, files in os.walk(topFolder):
            for filename in files:
                if IsImageFile(filename):
                    # You could store full paths if you prefer, but
                    # here we'll just store the filename.
                    existing_images.add(filename)

    return existing_images

def main():
    parser = argparse.ArgumentParser(description='Generate dithered dataset')
    parser.add_argument('--ditheringType', default='FloydSteinberg', type=str,
                        help='None, Riemersma or FloydSteinberg')
    options = parser.parse_args()

    ditheringType = options.ditheringType
    inputRootPath = 'raw training data'
    targetPathTemplate = 'data/{}/' + ditheringType
    dataTypes = ['train', 'val']
    minimumSize = 700 # this should be considerably (3x) larger than patchLength in datasetAugmentation.py
    upScaleFactor = 2

    # Precompute folder paths
    for dataType in dataTypes:
        targetPath = targetPathTemplate.format(dataType)
        referencePath = targetPath + '/target'
        transformedPath = targetPath + f'/data_{upScaleFactor}x'
        os.makedirs(referencePath, exist_ok=True)
        os.makedirs(transformedPath, exist_ok=True)

    # Build a dictionary/set of existing images to enable skipping
    existingImages = build_existing_image_dict(targetPathTemplate, dataTypes)

    # Iterate over input folders
    for inputFolderName in os.listdir(inputRootPath):
        inputFolderPath = os.path.join(inputRootPath, inputFolderName)
        inputFilenames = [x for x in listdir(inputFolderPath) if IsImageFile(x)]

        print(f'\nConverting folder {inputFolderName}...')
        # Iterate over images in the input folder
        for inputFilename in inputFilenames:
            print(f'\nConverting image {inputFilename}...')
            inputFilePath = os.path.join(inputFolderPath, inputFilename)
            base, ext = os.path.splitext(inputFilename)

            # 1) Quick check: is the original image too small?
            image = Image.open(inputFilePath)
            width, height = image.size
            if width < minimumSize or height < minimumSize:
                print('...image too small. Skipped.')
                continue

            # 2) Find all feasible downsampling scales for i in [0..8]
            feasible_scales = find_feasible_scales(
                width, height, upScaleFactor, minimumSize, num_attempts=9
            )
            if not feasible_scales:
                print('...image too small after random resizing. Skipped.')
                continue

            # 3) If we have at least one feasible scale, do the distance check
            distance = MeasureDistanceBetweenImageAndPalette(
                inputFilePath, 
                paletteFilePath='PerOxyd.bmp'
            )
            if distance > 8:
                print(f'...distance to palette: {distance:.1f}. Skipped.')
                continue

            # 4) We have one or more feasible scales, and distance is valid.
            #    Now decide whether each scale should go to train or val,
            #    and do the dithering.
            for (i, random_scale, transform_width, transform_height) in feasible_scales:
                # Decide whether this image pair belongs to train or val
                if random.random() > 0.8:
                    dataType = 'val'
                else:
                    dataType = 'train'

                dither_and_save(
                    inputFilePath,
                    baseFilename=base,
                    ditheringType=ditheringType,
                    upScaleFactor=upScaleFactor,
                    i=i,
                    transform_width=transform_width,
                    transform_height=transform_height,
                    dataType=dataType,
                    targetPathTemplate=targetPathTemplate,
                    existingImages=existingImages
                )

if __name__ == "__main__":
    main()