import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torchvision.models as models
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image


# download the pretrained model
import torchvision.models as mmodels
model = mmodels.resnet50(pretrained = True)

# switch device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data load
batch_size = 128
learning_rate = 0.0001
data_path = ''

image_transforms = {
    'train': transforms.Compose([
             transforms.Resize((128,128)),
             transforms.ToTensor()
    ]),
    'val': transforms.Compose([
             transforms.Resize((128,128)),
             transforms.ToTensor()
    ])
}

train_dataset = datasets.ImageFolder(data_path + '/train', transform=image_transforms['train'])
val_dataset = datasets.ImageFolder(data_path + '/val', transform=image_transforms['val'])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(train_dataset)
val_data_size = len(val_dataset)

# freeze the parameters
for param in model.parameters():
    param.requires_grad = False

#
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512,100)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(100,8)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.fc = fc


# shifting model to gpu
model.to(device)


# VALID FUNC
def validation(model, criterion, images, epoch):
    val_loss = 0.0
    val_acc = 0.0
    val_loss_arr = []
    val_acc_arr = []
    k = 0
    with torch.no_grad():
        model.eval()
        for j, (val_features, val_labels) in enumerate(val_dataloader):
            outputs = model(val_features)
            # Compute loss
            loss = criterion(outputs, val_labels)
            # Compute the total loss for the batch and add it to valid_loss
            val_loss += loss.item() * val_features.size(0)
            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(val_labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.sum(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to valid_acc
            val_acc += acc.item()

        print("Epoch: {:03d}, Val Loss: {:.4f}, Val Accuracy: {:.4f}".format(epoch, val_loss / val_data_size,
                                                                             val_acc / val_data_size))
        model.train()

    print('Finished Validation')


# TRAIN FUNC
def train(model, trainloader, criterion, optimizer, epochs=20):
    train_loss = []
    for e in range(epochs):
        running_loss = 0
        cnt = 0
        for images, labels in trainloader:
            inputs, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            img = model(inputs)

            loss = criterion(img, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            cnt += 1
            if cnt % 1 == 0:
                print(cnt, loss)
        print("Epoch : {}/{}..".format(e + 1, epochs), "Training Loss: {:.6f}".format(running_loss / len(trainloader)))
        train_loss.append(running_loss)
        validation(model, criterion, inputs, e)  # validation
        torch.save(model.state_dict(), 'resnet_fulldata_e' + str(e) + '.ptch')
    plt.plot(train_loss, label="Training Loss")
    plt.show()


# TRAINING
epochs = 20
model.train()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
criterion = nn.NLLLoss()
train(model,train_dataloader,criterion, optimizer, epochs)

