import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image


# INITIALIZATION OF NET
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, 7, padding=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.MaxPool2d(2, 2))
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(18, 36, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(36),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 55, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(55),
            nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(55, 72, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(72),
            nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(72 * 8 * 8, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 8))

    def forward(self, x):
        # print(f"Shape of tensor: {x.shape}")
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        # x = F.softmax(x, dim=1)
        # print(f"Shape of tensor: {x.shape}")
        return x


classifier = Net()
classifier.load_state_dict(torch.load('my_fds_e6+6_b160_s128_r_2_5.ptch'))

image_transforms = {  # transformation of input images
    'test': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
}


def predict(model, test_image_name):  # prediction function
    transform = image_transforms['test']
    test_image = Image.open(test_image_name).convert('RGB')
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 244, 244).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 128, 128)
    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        out = torch.exp(out)
        # out = F.softmax(out, dim=1)
        topk, topclass = out.topk(1, dim=1)
        print(test_image_name)
        print(f"Output class:  {topclass.cpu().numpy()[0][0] + 1}",
              "    prob :  {:.3f}".format(topk.cpu().numpy()[0][0]/torch.sum(out)), '     ', out, torch.sum(out))
    return topclass.cpu().numpy()[0][0] + 1


data_path = ''
results = open('results.txt', 'a')

classes = {1: 'Arabic', 2: 'Bangla', 3: 'Chinese', 4: 'Hindi', 5: 'Japanese', 6: 'Korean', 7: 'Latin', 8: 'Symbols', }

count = 0
for image in os.listdir(data_path):
    print(count)
    count += 1
    res = predict(classifier, data_path + image)
    results.write(image + ',' + classes[res] + '\n')
