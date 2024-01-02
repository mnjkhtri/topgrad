#in pytorch
import argparse
import io
import requests
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

torch.set_printoptions(precision=2, sci_mode=False)

class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: 
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, num, num_classes=None):
        super().__init__()
        self.num = num
        self.block = {
            34: BasicBlock,
        }[num]

        self.num_blocks = {
            34: [3, 4, 6, 3]
        }[num]

        self.in_planes = 64
        # -> (224 x 224) | 3
        
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # -> (114 x 114) | 64

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)# -> (56 x 56) | 64
        self.layer1 = self._make_layer(self.block, 64,  self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # -> (1 x 1) | 512

        self.fc = nn.Linear(512 * self.block.expansion, num_classes) if num_classes is not None else None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if block == BasicBlock:
                layers.append(block(self.in_planes, planes, stride))
            else:
                raise NotImplementedError
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def load_from_pretrained(self):
        url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        pretrained = torch.hub.load_state_dict_from_url(url)
        self.load_state_dict(pretrained, strict=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference using ResNet model with an image URL")
    parser.add_argument("image_url", help="URL of the image to perform inference on")
    args = parser.parse_args()

    import requests
    response = requests.get(args.image_url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),             
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = preprocess(img)
    img = img.unsqueeze(0)

    model = ResNet(34, 1000)
    model.load_from_pretrained()
    with torch.no_grad():
        model.eval()
        logits = model(img)
    index = torch.argmax(F.softmax(logits, dim=-1))

    with open('./imagenet1000_clsidx_to_labels.txt') as f:
        labels = eval(f.read())
        cat = labels.get(index.item(), "Label not found")
    print("The image is of", cat)
