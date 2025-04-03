import os
import random
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset


def IsImageFile(filename):
    """
    Check if a file has a valid image extension.
    """
    validExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    base, ext = os.path.splitext(filename)
    return ext.lower() in validExtensions


def CreateShadowMasks(transPatchWidth, transPatchHeight, upScale=2, transShadowLength=8):
    """
    Create a pair of drop-shadow masks:
      1) A 50%-opacity black shadow for the reference image (full resolution).
      2) A 1-pixel checkerboard black/transparent pattern for the transformed image (downscaled).

    The shadow is 'cast' along a 45° diagonal from the patch bounds.
    """

    # The shadow extends from the top-left corner of the patch diagonally down-right.
    # We need a bounding box that can contain the patch plus the maximum diagonal offset.
    refPatchHeight = transPatchHeight * upScale
    refPatchWidth = transPatchWidth * upScale
    refShadowLength = transShadowLength * upScale

    # Create boolean mask for REF scale via slicing
    refBoxWidth = refPatchWidth + refShadowLength
    refBoxHeight = refPatchHeight + refShadowLength
    refShadowMask = np.zeros((refBoxHeight, refBoxWidth), dtype=bool)
    # For each offset d in [0..refShadowLength), shift patch area by (d, d).
    for d in range(refShadowLength):
        top = d
        left = d
        bottom = d + refPatchHeight
        right = d + refPatchWidth
        refShadowMask[top:bottom, left:right] = True

    # Build the reference RGBA shadow (50% opacity black)
    refShadowArray = np.zeros((refBoxHeight, refBoxWidth, 4), dtype=np.uint8)
    refShadowArray[refShadowMask, 3] = 128 # alpha channel = 128 where mask is True
    refShadowImage = Image.fromarray(refShadowArray, mode='RGBA')

    # Create boolean mask for TRANS scale via slicing
    transBoxHeight = transPatchHeight + transShadowLength
    transBoxWidth = transPatchWidth + transShadowLength
    transShadowMask = np.zeros((transBoxHeight, transBoxWidth), dtype=bool)
    for d in range(transShadowLength):
        top = d
        left = d
        bottom = d + transPatchHeight
        right = d + transPatchWidth
        transShadowMask[top:bottom, left:right] = True

    # Build the checkerboard pattern
    yy, xx = np.indices(transShadowMask.shape)
    checkerboard = ((xx + yy) % 2 == 1)  # True will become black, False will become transparent

    # Combine checkerboard pattern and transformed mask
    blackLocations = checkerboard & transShadowMask
    transShadowArray = np.zeros((transBoxHeight, transBoxWidth, 4), dtype=np.uint8)
    transShadowArray[blackLocations, 3] = 255
    transShadowImage = Image.fromarray(transShadowArray, mode='RGBA')

    return refShadowImage, transShadowImage


class DatasetFromFolder(Dataset):
    """
    Loads pre-rendered data (dithered + reference pairs).
    Then applies random augmentation: random patches, shape overlays, etc.
    """
    def __init__(self, datasetDir, ditheringType):
        super(DatasetFromFolder, self).__init__()
        self.upScale = 2
        self.imageDir = f'{datasetDir}/{ditheringType}/data_{self.upScale}x'
        self.targetDir = f'{datasetDir}/{ditheringType}/target'
        
        self.imageFilenames = [
            join(self.imageDir, x)
            for x in listdir(self.imageDir) if IsImageFile(x)
        ]
        self.targetFilenames = [
            join(self.targetDir, x)
            for x in listdir(self.imageDir) if IsImageFile(x)
        ]
        
        self.inputTransform = transforms.ToTensor()
        self.targetTransform = transforms.ToTensor()
        self.imagePatches = []
        self.targetPatches = []
        self.patchCount = 0

    def __getitem__(self, index):
        imageFilename = self.imageFilenames[index]
        targetFilename = self.targetFilenames[index]
        
        image = Image.open(imageFilename).convert('RGB')
        target = Image.open(targetFilename).convert('RGB')

        up = self.upScale
        patchLength = 96  # reference length of a training square
        heightCrop, widthCrop = (patchLength // up, patchLength // up)
        xSize, ySize = image.size
        
        # Extract a random square patch from the two source images and define it as background
        randX = random.randint(0, (xSize - heightCrop) - 1)
        randY = random.randint(0, (ySize - widthCrop) - 1)
        imageBg = TF.crop(image, randY, randX, heightCrop, widthCrop)
        targetBg = TF.crop(target, randY * up, randX * up, heightCrop * up, widthCrop * up)

        # Retrieve a different, smaller image patch
        if (self.patchCount < 100 and random.random() < 0.1) or self.patchCount == 0:
            # If the patch database is not full yet, retrieve it from the source image directly
            hPatch = random.randint(1, heightCrop // 4) * 2 # patch sizes need to be even for the checkerbox to work out
            wPatch = random.randint(1, widthCrop // 4) * 2
            patchX = random.randint(0, (xSize - hPatch) - 1)
            patchY = random.randint(0, (ySize - wPatch) - 1)
            
            imagePatch = TF.crop(image, patchY, patchX, hPatch, wPatch)
            targetPatch = TF.crop(target, patchY * up, patchX * up, hPatch * up, wPatch * up)
            
            self.imagePatches.append(imagePatch)
            self.targetPatches.append(targetPatch)
            self.patchCount += 1
        else:
            # Retrieve it from the patch database
            patchID = random.randint(0, self.patchCount - 1)
            imagePatch = self.imagePatches[patchID]
            targetPatch = self.targetPatches[patchID]
            wPatch, hPatch = imagePatch.size

        # Render two drop shadows
        shadowRef, shadowTrans = CreateShadowMasks(wPatch, hPatch, upScale=up, transShadowLength=random.randint(4, 8))

        # Pick a random location for the new patch (and its shadow)
        randX = random.randint(0, (widthCrop - wPatch) - 1)
        randY = random.randint(0, (heightCrop - hPatch) - 1)

        shadowPlaced = False
        if random.random() < 0.2 and False:
            # Place the shadow into the reference image
            targetBgAlpha = targetBg.convert('RGBA')
            targetBgAlpha.paste(shadowRef, (randX * up, randY * up), shadowRef)
            targetBg = targetBgAlpha.convert('RGB')

            # Paste the downscaled checkerboard shadow into the transformed image
            imageBgAlpha = imageBg.convert('RGBA')
            imageBgAlpha.paste(shadowTrans, (randX, randY), shadowTrans)
            imageBg = imageBgAlpha.convert('RGB')
            shadowPlaced = True

        # Place the new patch on top of each shadow
        if random.random() < 0.8 or shadowPlaced:
            imageBg.paste(imagePatch, (randX, randY))
            targetBg.paste(targetPatch, (randX * up, randY * up))

        # Occasionally, save out a training sample
        if random.randint(1, 500) == 42:
            sampleFilename = f'training samples/{index:04d}.png'
            if not os.path.isfile(sampleFilename):
                sampleImg = Image.new('RGB', (heightCrop * 2 * up + 1, widthCrop * up))
                
                if up > 1:
                    imageResized = imageBg.resize((heightCrop * up, widthCrop * up), Image.NEAREST)
                else:
                    imageResized = imageBg
                
                sampleImg.paste(imageResized, (0, 0))
                sampleImg.paste(targetBg, (heightCrop * up + 1, 0))
                sampleImg.save(sampleFilename)

        # Convert final backgrounds to Tensors
        imageTensor = self.inputTransform(imageBg)
        targetTensor = self.targetTransform(targetBg)
        
        return imageTensor, targetTensor

    def __len__(self):
        return len(self.imageFilenames)