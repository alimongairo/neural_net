import torch
import cv2
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = cv2.dnn.readNet('frozen_east_text_detection.pb')
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
        topk, topclass = out.topk(1, dim=1)
        print(f"Output class:  {topclass.cpu().numpy()[0][0] + 1}",
              "    prob :  {:.3f}".format(topk.cpu().numpy()[0][0]), '     ', out, torch.sum(out))
    return topclass.cpu().numpy()[0][0] + 1


data_path = ''
results = open('mynet_results.txt', 'a')

classes = {1: 'Arabic', 2: 'Bangla', 3: 'Chinese', 4: 'Hindi', 5: 'Japanese', 6: 'Korean', 7: 'Latin', 8: 'Symbols'}


def detector(img_name, class_dict):
    img = cv2.imread(img_name)  # импорт картинки
    orig = img.copy()
    rgb = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    (Wi, He) = rgb.shape[:2]

    small_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) # установка цветового пространства RGB, серый фильтр

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # ядро элиптической формы 3х3
    grad = cv2.morphologyEx(small_img, cv2.MORPH_GRADIENT, kernel)  # оставляем только границы

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # определение порога ?

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))  # прямоугольное ядро 9х1
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel) # заполнение пропусков в контурах

    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # вычисление контуров

    tmp_str = ''
    tmp_img = 'tmp.jpg'
    mask = np.zeros(bw.shape, dtype=np.uint8)  # выделение области
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if 0.5 < r < 0.9 and w > Wi * 0.01 and h > He * 0.01:
            x *= 2
            y *= 2
            w *= 2
            h *= 2
            tmp_str += str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(y) + ',' + str(x + w) + ',' + str(
                y + h) + ',' + str(x) + ',' + str(y + h) + ',0.7 ,'
            cropped = orig[y:y + h, x:x + w]
            cv2.imwrite(tmp_img, cropped)
            res = predict(classifier, tmp_img)
            tmp_str += class_dict[res] + '\n'
    return tmp_str


data_path = ''
count = 1

for image in os.listdir(data_path):
    print(count)
    count += 1
    id_list = image.split('_')
    img_id = id_list[2].split('.')
    result = open('res_img_' + img_id[0] + '.txt', 'w')
    res_str = detector(data_path + image, classes)
    result.write(res_str)
