from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc1 = nn.Linear(1000, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc1(h)
        h = self.fc2(h)
        return h




#transform = transforms.Compose([
    #transforms.Resize(100, interpolation=InterpolationMode.LANCZOS),
    #transforms.ToTensor()
#])

#class Net(pl.LightningModule):

    #def __init__(self):
        #super().__init__()

        #self.conv = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, padding = 1)
        #self.bn = nn.BatchNorm2d(6)
        #self.fc = nn.Linear(6*50*50, 2)


    #def forward(self, x):

        #h = self.conv(x)
        #h = F.relu(h)
        #h = self.bn(h)
        #h = F.max_pool2d(h, kernel_size=2, stride=2)
        #h = h.view(-1, 6*50*50)
        #h = self.fc(h)
        #return h
    
    
    
