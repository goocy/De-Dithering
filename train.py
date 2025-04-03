import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.engine import Engine

from datasetAugmentation import DatasetFromFolder
from model import Net
from psnrmeter import PSNRMeter


def Processor(sample):
    data, target, training = sample
    data = Variable(data)
    target = Variable(target)
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    output = model(data)
    loss = criterion(output, target)
    return loss, output


def OnSample(state):
    state['sample'].append(state['train'])


def ResetMeters():
    meterPSNR.reset()
    meterLoss.reset()


def OnForward(state):
    meterPSNR.add(state['output'].data, state['sample'][1])
    meterLoss.add(state['loss'].item())


def OnStartEpoch(state):
    ResetMeters()
    optimizer.step()
    scheduler.step()
    state['iterator'] = tqdm(state['iterator'])


def OnEndEpoch(state):
    print('[Epoch %d] Train Loss: %.4f (PSNR: %.2f dB)' %
          (state['epoch'], meterLoss.value()[0], meterPSNR.value()))
    ResetMeters()
    engine.test(Processor, valLoader)
    print('[Epoch %d] Val Loss: %.4f (PSNR: %.2f dB)' %
          (state['epoch'], meterLoss.value()[0], meterPSNR.value()))

    torch.save(model.state_dict(), 'epochs/epoch_%s_%d.pt' % (ditheringType, state['epoch']))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training the de-dithering network')
    parser.add_argument('--ditheringType', default='FloydSteinberg', type=str,
                        help='None, Riemersma or FloydSteinberg')
    parser.add_argument('--numEpochs', default=250, type=int, help='number of epochs')
    options = parser.parse_args()

    ditheringType = options.ditheringType

    trainSet = DatasetFromFolder('data/train', ditheringType=ditheringType)
    valSet = DatasetFromFolder('data/val', ditheringType=ditheringType)
    trainLoader = DataLoader(dataset=trainSet, num_workers=6, batch_size=40, shuffle=True)
    valLoader = DataLoader(dataset=valSet, num_workers=6, batch_size=40, shuffle=False)

    model = Net()
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    engine = Engine()
    meterLoss = tnt.meter.AverageValueMeter()
    meterPSNR = PSNRMeter()

    engine.hooks['on_sample'] = OnSample
    engine.hooks['on_forward'] = OnForward
    engine.hooks['on_start_epoch'] = OnStartEpoch
    engine.hooks['on_end_epoch'] = OnEndEpoch

    engine.train(Processor, trainLoader, maxepoch=options.numEpochs, optimizer=optimizer)