import torch
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
        input = self.fc(input)
        input = self.softmax(self.relu(input))
        return input