import torch
import cv2
from PIL import Image
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import os


def east(img_name):
    i = 0
    tmp_str = ''
    # switch device to gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    # load the input image and grab the image dimensions
    image = cv2.imread(img_name)
    orig = image.copy()
    (H, W) = image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    width = 320
    height = 320
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    min_confidence = 0.5

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(img_rgb, 1.0, (W, H), (123.68, 116.78, 103.94), True, False)
    print(layerNames)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    # show timing information on text prediction
    # print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    print(rects)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    crop_number = 0
    print(orig.shape[:2])

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        print(startX, startY, endX, endY)
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        print(startX, startY, endX, endY)
        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cropped = orig[startY:endY, startX:endX]
        tmp_str += str(startX) + ',' + str(startY) + ',' + str(endX) + ',' + str(startY) + ',' + str(
            endX) + ',' + str(endY) + ',' + str(startX) + ',' + str(endY) + ',' + str(int(confidences[i] * 100) / 100) + '\n'
        i += 1
    return tmp_str


data_path = ''
count = 1
res_str = ''
for image in os.listdir(data_path):
    print(count)
    count += 1
    id_list = image.split('_')
    id = id_list[2].split('.')
    result = open('res_img_' + id[0] + '.txt', 'w')
    res_str = east(data_path + image)
    result.write(res_str)

