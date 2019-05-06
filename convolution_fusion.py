import dataloader
import os
import torch
from torch.backends import cudnn
from flow_train_partial import FLOW_Inception
from rgb_train_partial import RGB_Inception



os.environ["CUDA_VISIBLE_DEVICES"] = "6"
cudnn.benchmark = True

"""
(conv1): 
        Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
"""

class ConvFused(torch.nn.Module):
    def __init__(self):
        super(ConvFused, self).__init__()
        self.conv31= torch.nn.Conv3d(1024,512,kernel_size=1, stride = 1, bias = False)
        self.bn31= torch.nn.BatchNorm3d(512)
        self.conv32= torch.nn.Conv3d(512,512,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn32= torch.nn.BatchNorm3d(512)
        self.conv33= torch.nn.Conv3d(512, 1024,kernel_size=1, stride=1, bias=False)
        self.bn33= torch.nn.BatchNorm3d(1024)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool3d(kernel_size=(2,1,1), stride=(2,1,1),padding=0)
        self.fc = torch.nn.Linear(1024,63)
        self.softmax = torch.nn.Softmax()

    def forward(self, input):
        input = self.conv31(input)
        input = self.bn31(input)
        input = self.conv32(input)
        input = self.bn32(input)
        input = self.conv33(input)
        input = self.bn33(input)
        input = self.relu(input)
        input = self.avgpool(input)
        input = input.view(input.size(0),-1)
        input = self.fc(input.flatten())
        input = self.softmax(self.relu(input))
        return input



motion_loader = dataloader.EgoCentricDataLoader(
    BATCH_SIZE=1,
    num_workers=8,
    img_stack=10,
    train_list_path='/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_train_list.txt',
    test_list_path='/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_test_list.txt',
    val_list_path='/home/Students/k_b459/Projects/360twostreamv2/360twostream/Egok_list/merged_val_list.txt',
    mode='both',
    onlytest=True,
    # onlyval = True,
    pin_memory=False
)


testLoader = motion_loader()

for i, ((d1, d2), l) in enumerate(testLoader):
    break


MCNN = FLOW_Inception(
    nb_epochs=None,
    lr=1e-2,
    batch_size=None,
    resume='record_partial/flow/model_best.pth.tar',
    start_epoch=None,
    evaluate=None,
    train_loader=None,
    val_loader=None,
    # channel=20,
    test_loader=None
    )

SCNN = RGB_Inception(
    nb_epochs=None,
    lr=1e-2,
    batch_size=None,
    resume='record_partial/rgb/model_best.pth.tar',
    start_epoch=None,
    evaluate=None,
    train_loader=None,
    test_loader=None,
    val_loader=None,
)


MCNN.load_only()
SCNN.load_only()
#
# mcnn_layers = list(
#     MCNN.model._modules.keys())  # ['conv1_custom','bn1','relu','maxpool','layer1','layer2','layer3','layer4','avgpool','fc_custom']
# scnn_layers = list(
#     SCNN.model._modules.keys())  # ['conv1_custom','bn1','relu','maxpool','layer1','layer2','layer3','layer4','avgpool','fc_custom']
#
# extract = lambda model,x:model.model._modules.get(x)
#
# mcnn_head = torch.nn.Sequential(*map(lambda x:extract(MCNN,x),mcnn_layers[:-2]))##:4 upto maxpool
# scnn_head = torch.nn.Sequential(*map(lambda x:extract(SCNN,x),scnn_layers[:-2]))##:4 upto maxpool
#
# mcnn_head.cuda(1)
# scnn_head.cuda(2)

for param_mcnn in MCNN.model.parameters():
    param_mcnn.requires_grad = False

for param_scnn in SCNN.model.parameters():
    param_scnn.requires_grad = False


d1.cuda()
d2.cuda()

_,(_,mcnnout) = MCNN.extractProbability(d1)
_,(_,scnnout) = SCNN.extractProbability(d2)

mcnnout.cuda()
scnnout.cuda()

print('mcnn_head_out',mcnnout.shape)
print('scnn_head_out',scnnout.shape)

combo = torch.cat([mcnnout,scnnout],dim = 2)#torch.Size([1, 2048, 2, 7, 7])
combo.cuda()
# combo.cuda()
c = ConvFused()
c.cuda()

out = c(combo)

print(out.shape)

"""
(conv1): 
        Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
"""