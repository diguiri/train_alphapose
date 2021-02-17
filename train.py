
# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
import torch
import torch.utils.data
from utils.dataset import coco
from opt import opt
from tqdm import tqdm

from utils.eval import DataLogger, accuracy
from utils.img import flip, shuffleLR
from evaluation import prediction
import torch.nn as nn
import torch
from models.layers.DUC import DUC
from models.layers.SE_Resnet import SEResnet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# Import training option
#from opt import opt
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import os

def createModel():
    return FastPose_SE()


class FastPose_SE(nn.Module):
    conv_dim = 128

    def __init__(self):
        super(FastPose_SE, self).__init__()

        self.preact = SEResnet('resnet101')

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.conv_dim, 33, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out
    
    
def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    train_loader_desc = tqdm(train_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(train_loader_desc):
        inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        setMask = setMask.cuda()
        out = m(inps).narrow(1,0,17)
       

        loss = criterion(out.mul(setMask), labels)

        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar(
            'Train/Loss', lossLogger.avg, opt.trainIters)
        writer.add_scalar(
            'Train/Acc', accLogger.avg, opt.trainIters)

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(val_loader_desc):
        inps = inps.cuda()
        labels = labels.cuda()
        setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            loss = criterion(out.mul(setMask), labels)

            flip_out = m(flip(inps))
            flip_out = flip(shuffleLR(flip_out, val_loader.dataset))

            out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar(
            'Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar(
            'Valid/Acc', accLogger.avg, opt.valIters)

        val_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def main():

    # Model Initialize
    m = createModel().cuda()
    m.load_state_dict(torch.load('duc_se.pth'))

    criterion = torch.nn.MSELoss().cuda()

    if opt.optMethod == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(),
                                        lr=opt.LR,
                                        momentum=opt.momentum,
                                        weight_decay=opt.weightDecay)
    elif opt.optMethod == 'adam':
        optimizer = torch.optim.Adam(
            m.parameters(),
            lr=opt.LR
        )
    else:
        raise Exception

    writer = SummaryWriter(
        '.tensorboard/{}/{}'.format(opt.dataset, opt.expID))

    # Prepare Dataset
    if opt.dataset == 'coco':
        train_dataset = coco.Mscoco(train=True)
        val_dataset = coco.Mscoco(train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.trainBatch, shuffle=True, num_workers=opt.nThreads, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.validBatch, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

    # Model Transfer
    m = torch.nn.DataParallel(m).cuda()

    # Start Training
    for i in range(opt.nEpochs):
        opt.epoch = i

        print('############# Starting Epoch {} #############'.format(opt.epoch))
        loss, acc = train(train_loader, m, criterion, optimizer, writer)

        print('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=opt.epoch,
            loss=loss,
            acc=acc
        ))

        opt.acc = acc
        opt.loss = loss
        m_dev = m.module
        if i % opt.snapshot == 0:
            torch.save(
                m_dev.state_dict(), '../exp/{}/{}/model_{}.pkl'.format(opt.dataset, opt.expID, opt.epoch))
            torch.save(
                opt, '../exp/{}/{}/option.pkl'.format(opt.dataset, opt.expID, opt.epoch))
            torch.save(
                optimizer, '../exp/{}/{}/optimizer.pkl'.format(opt.dataset, opt.expID))

        loss, acc = valid(val_loader, m, criterion, optimizer, writer)

        print('Valid-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))

       
    writer.close()


if __name__ == '__main__':
    main()
