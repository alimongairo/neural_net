import cv2
import numpy as np
import os
import torchvision.transforms as transforms


def detector(img_name):
    img = cv2.imread(img_name)  # импорт картинки
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
    mask = np.zeros(bw.shape, dtype=np.uint8)  # выделение области
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if 0.6 < r < 0.8 and w > Wi * 0.05 and h > He * 0.05:
            tmp_str += str(x*2) + ',' + str(y*2) + ',' + str(x*2 + w*2) + ',' + str(y*2) + ',' + str(x*2 + w*2) + ',' + str(
                y*2 + h*2) + ',' + str(x*2) + ',' + str(y*2 + h*2) + ',0.6\n'
    return tmp_str


data_path = ''
count = 1
for image in os.listdir(data_path):
    print(count)
    count += 1
    id_list = image.split('_')
    id = id_list[2].split('.')
    result = open('' + id[0] + '.txt', 'w')
    res_str = detector(data_path + image)
    result.write(res_str)
