from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

alexnet = models.alexnet(pretrained=True)
print(alexnet)

class AlexNet(nn.Module):
    # input images: 227x227x3
    def __init__(self, nb_class=1000):
        super(AlexNet, self).__init__()
        # 3 input image channel, 96 output channels, 11x11 square convolution
        self.features = nn.Sequential(
            # 3x
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # 64x
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            #
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256x6x6
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, nb_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # x.size(0) = batch size ?
        x = self.classifier(x)
        return x

# test_alexnet = AlexNet()
# print(test_alexnet)

def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

# model_alexnet = alexnet(pretrained=True)
# print(model_alexnet)