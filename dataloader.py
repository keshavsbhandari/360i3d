from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

get_video_name = lambda x: x.split('/')[-2]


# keys = ['images','flowu','flowv','video','class','parentclass','label']

class egocentric_dataset(Dataset):

    def __init__(self,
                 data_list_dict,
                 img_stack=10,
                 transform=None,
                 rows=320,
                 cols=640,
                 resizerow=224,
                 resizecol=224,
                 num_classes=63,
                 istest=False,
                 mode='motion'
                 ):
        self.data = data_list_dict
        self.transform = transform
        self.img_stack = img_stack
        self.img_rows = rows
        self.img_cols = cols
        self.resizerow = resizerow
        self.resizecol = resizecol
        self.num_classes = num_classes
        self.istest = istest
        self.mode = mode

    def stackopf(self, data):
        #flow = torch.FloatTensor(2 * self.img_stack, self.resizerow, self.resizecol)
        d = []
        for i, (u, v) in enumerate(zip(data['flowu'], data['flowv'])):
            U = self.transform(Image.fromarray(cv2.imread(u, cv2.IMREAD_GRAYSCALE)))
            V = self.transform(Image.fromarray(cv2.imread(v, cv2.IMREAD_GRAYSCALE)))
            d.append(torch.stack([U[0],V[0]]))
            #flow[2 * i, :, :] = U
            #flow[2 * i + 1, :, :] = V
        return torch.stack(d, dim = 1)
        #return flow

    def __len__(self):
        return len(self.data)

    def getimages(self, data):
        #imgs = torch.FloatTensor(3 * self.img_stack, self.resizerow, self.resizecol)
        d = []
        for i, img_path in enumerate(data['images']):
            #image = self.transform(Image.fromarray(cv2.imread(img_path)))
            d.append(self.transform(Image.open(img_path)))
            #imgs[3 * i, :, :] = image[0]
            #imgs[3 * i + 1, :, :] = image[1]
            #imgs[3 * i + 1 + 1, :, :] = image[2]
        return torch.stack(d, dim = 1)
        #return imgs

    def get_both(self,data):
        flows = self.stackopf(data)
        imgs = self.getimages(data)
        return flows, imgs

    def __getitem__(self, idx):
        data = self.data[idx]
        label = {'class': data['class'],
                 'label': data['label'],
                 'superclass': data['parentclass'],
                 'vidname': data['video']}

        sample = {'motion': self.stackopf, 'spatial': self.getimages, 'both':self.get_both}.get(self.mode)(data)
        return sample, label
        #return [*{'motion': self.stackopf, 'spatial': self.getimages, 'both':self.get_both}.get(self.mode)(data),label]


class EgoCentricDataLoader():
    def __init__(self,
                 BATCH_SIZE,
                 num_workers,
                 train_list_path,
                 test_list_path,
                 val_list_path,
                 img_stack=10,
                 mode='motion',
                 onlytest = False,
                 onlyval = False,
                 pin_memory = True):

        self.BATCH_SIZE = BATCH_SIZE
        self.num_workers = num_workers
        self.img_stack = img_stack
        self.train_list = self.read_list(train_list_path)
        self.test_list = self.read_list(test_list_path)
        self.val_list = self.read_list(val_list_path)
        self.onlytest = onlytest
        self.pin_memory = pin_memory
        self.onlyval = onlyval

        random.shuffle(self.train_list)
        random.shuffle(self.test_list)
        random.shuffle(self.val_list)

        self.train_list = self.train_list
        self.val_list = self.val_list
        self.test_list = self.test_list

        self.mode = mode

    def __call__(self, *args, **kwargs):
        if self.onlytest:
            return self.load(data_list=self.test_list, istest=True)

        if self.onlyval:
            return self.load(data_list=self.val_list, istest=True)

        return self.load(data_list=self.train_list), \
               self.load(data_list=self.val_list, isval=True), \
               self.load(data_list=self.test_list, istest=True)

    def read_list(self, flowlist_path):
        with open(flowlist_path, 'r') as f:
            x = json.loads(f.read())
        return x

    def load(self, data_list, istest=False, isval=False):
        dataset = egocentric_dataset(
            data_list_dict=data_list,
            img_stack=self.img_stack,
            transform=transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]),
            resizecol=224,
            resizerow=224,

            istest=istest,
            mode=self.mode,
        )
        if istest:
            print('==> Testing data :', len(dataset))
        elif isval:
            print('==> Validation data :', len(dataset))
        else:
            print('==> Training data :', len(dataset))

        return DataLoader(
            dataset=dataset,
            batch_size={True: 1, False: self.BATCH_SIZE}.get(istest or isval),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


if __name__ == '__main__':
    data_loader = EgoCentricDataLoader(
        BATCH_SIZE=10,
        num_workers=1,
        img_stack=10,
        train_list_path= '/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_train_list.txt', 
        test_list_path='/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_test_list.txt',
        val_list_path='/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_val_list.txt',
        mode='spatial'
    )
    flow_train_loader, flow_val_loader, flow_test_loader, = data_loader()

    for i, (f, fl) in enumerate(flow_train_loader):
        break

    for i, (ftest, ftestl) in enumerate(flow_test_loader):
        break

    for i, (fval, fvall) in enumerate(flow_val_loader):
        break
