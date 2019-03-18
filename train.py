import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from tqdm import tqdm

from prep_training_data import DatasetFromFolder
from model import Net
from psnrmeter import PSNRMeter


def processor(sample):
    data, target, training = sample
    data = Variable(data)
    target = Variable(target)
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    output = model(data)#.permute(0,3,1,2)
    loss = criterion(output, target)

    return loss, output


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_psnr.reset()
    meter_loss.reset()


def on_forward(state):
    meter_psnr.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    scheduler.step()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Train Loss: %.4f (PSNR: %.2f db)' % (state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    reset_meters()

    engine.test(processor, val_loader)
    
    print('[Epoch %d] Val Loss: %.4f (PSNR: %.2f db)' % (state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    torch.save(model.state_dict(), 'epochs/epoch_%s_%d.pt' % (dithering_type, state['epoch']))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training the de-dithering network')
    parser.add_argument('--dithering_type', default='FloydSteinberg', type=str, help='None, Riemersma or FloydSteinberg')
    parser.add_argument('--num_epochs', default=250, type=int, help='number of epochs')
    opt = parser.parse_args()

    dithering_type = opt.dithering_type
    NUM_EPOCHS = opt.num_epochs

    train_set = DatasetFromFolder('data/train', dithering_type=dithering_type)
    val_set = DatasetFromFolder('data/val', dithering_type=dithering_type)
    train_loader = DataLoader(dataset=train_set, num_workers=5, batch_size=40, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=5, batch_size=40, shuffle=False)

    model = Net()
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    #model.load_state_dict(torch.load('epochs/Schritt 15.pt'))
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
