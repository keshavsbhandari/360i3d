from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import os
from glob import glob
from os.path import join
from itertools import chain
from tabulate import tabulate
import random
from datautils import savefile, openfile
from pathlib import Path
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class EgoCentricI3d(Dataset):
    def __init__(self,
                 data_list=None,
                 depth=100,
                 transforms=None,
                 mode='flow',
                 ):

        self.data_list = data_list
        self.depth = depth
        self.transform = transforms
        self.mode = mode

    def repeat_to_depth(self, framelist):
        mult = 1
        if len(framelist)<100:
            mult = 100//len(framelist) + 1
        framelist = framelist * mult
        return framelist[:100]

    def get_frames(self, data):
        frames = {}

        if self.mode == 'rgb':
            frames['rgb'] = self.repeat_to_depth(sorted(glob(join(data['imgs'], '*.jpg')), key=lambda x: int(Path(x).stem)))
        elif self.mode == 'flow':
            frames['u'] = self.repeat_to_depth(sorted(glob(join(data['u'], '*.jpg')), key=lambda x: int(Path(x).stem)))
            frames['v'] = self.repeat_to_depth(sorted(glob(join(data['v'], '*.jpg')), key=lambda x: int(Path(x).stem)))
        elif self.mode == 'both':
            frames['rgb'] = self.repeat_to_depth(sorted(glob(join(data['imgs'], '*.jpg')), key=lambda x: int(Path(x).stem)))
            frames['u'] = self.repeat_to_depth(sorted(glob(join(data['u'], '*.jpg')), key=lambda x: int(Path(x).stem)))
            frames['v'] = self.repeat_to_depth(sorted(glob(join(data['v'], '*.jpg')), key=lambda x: int(Path(x).stem)))
        else:
            raise Exception("Error data mode")

        self.frames = frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        self.get_frames(data)
        label = {'class': data['cls'],
                 'label': data['label'],
                 'superclass': data['superclass'],
                 'vidname': data['vidname']}
        return {'flow': self.getflows, 'rgb': self.getrgbs, 'both': self.get_both}.get(self.mode)(label)
        # return out,label
    def get_both(self,label):
        return self.getflows(label)[0],self.getrgbs(label)[0], label

    def getrgbs(self, label):
        d = []
        for frame in self.frames['rgb']:
            d.append(self.transform(Image.open(frame)))
        return torch.stack(d, dim = 1),label

    def getflows(self,label):
        d = []
        for u,v in zip(self.frames['u'],self.frames['v']):
            U = self.transform(Image.fromarray(cv2.imread(u, cv2.IMREAD_GRAYSCALE)))
            V = self.transform(Image.fromarray(cv2.imread(v, cv2.IMREAD_GRAYSCALE)))
            d.append(torch.stack([U[0],V[0]]))

        return torch.stack(d, dim = 1), label


class EgoCentricI3dLoader():
    def __init__(self, BATCH_SIZE, num_workers, train_list, test_list, val_list, mode, depth, onlytest=False,
                 onlyval=False):
        self.batch_size = BATCH_SIZE
        self.num_workers = num_workers
        self.train_list = openfile(train_list)
        self.test_list = openfile(test_list)
        self.val_list = openfile(val_list)
        self.mode = mode
        self.depth = depth
        self.onlytest = onlytest
        self.onlyval = onlyval

        assert self.mode in ['flow', 'rgb', 'both'], "mode should be in 'rgb' or 'flow' or both"

    def __call__(self, *args, **kwargs):
        if self.onlytest: return load(data_list=self.test_list, istest=True)
        if self.onlyval: return load(data_list=self.val_list, isval=True)
        return self.load(data_list=self.train_list), \
               self.load(data_list=self.val_list, isval=True), \
               self.load(data_list=self.test_list, istest=True)

    def load(self, data_list, istest=False, isval=False):
        dataset = EgoCentricI3d(
            data_list=data_list,
            depth=self.depth,
            transforms=transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor()
            ]),
            mode=self.mode
        )

        if istest:
            print('====> Testing data : ', len(dataset))
        elif isval:
            print('====> Validation data : ', len(dataset))
        else:
            print('====> Training data : ', len(dataset))

        return DataLoader(
            dataset=dataset,
            batch_size={True: 1, False: self.batch_size}.get(istest or isval),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == '__main__':
    dataloader = EgoCentricI3dLoader(
        BATCH_SIZE=10,
        num_workers=8,
        train_list='./data_list/train.txt',
        test_list='./data_list/test.txt',
        val_list='./data_list/val.txt',
        mode='rgb',
        depth=100
    )

    trainx, valx, testx = dataloader()

    for ind, (i,l) in enumerate(trainx): break

    for iind, (ii,ll) in enumerate(testx):break

