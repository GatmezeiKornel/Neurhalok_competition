import csv
import os
import numpy as np
import time
import torch
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import pathlib
import math


class VGG(nn.Module):
    def __init__(self, vgg_name):
        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        super(VGG, self).__init__()
        self.features = self._make_layers(self.cfg[vgg_name])
        self.classifier = nn.Linear(512, 8)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    # This method will generate us all of layers
    # of VGG according to the cfg which now is
    # a list of numbers and 'M' charachters
    def _make_layers(self, cfg):
        layers = []
        in_channels = 2
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)


class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)

        self.fc = nn.Linear(in_features=32 * 64 * 64, out_features=64*64)
        self.fc2 = nn.Linear(in_features=64*64, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=128)

        self.fco = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)

        output = self.bn1(output)
        output = self.relu(output)
        output = self.pool(output)
        output = nn.Dropout(0.4)(output)

        output = self.conv2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu(output)
        output = torch.flatten(output, 1)
        output = output.view(-1, 32 * 64 * 64)
        output = self.fc(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fco(output)


        return output


def preprocess():
    transformer = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]
                             )
    ])
    transformer2 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]
                             )
    ])
    return transformer, transformer2


def dataLoader(train_path, test_path, transformer, transformer2=None):
    if transformer2 is None:
        transformer2 = transformer

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transformer)
        , batch_size=256, shuffle=True
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transformer2)
        # , batch_size=256, shuffle=True
    )
    return train_loader, test_loader


def loadCategories(path):
    root = pathlib.Path(path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    return classes


def saveList(listToSave):
    with open('submission.csv', 'w') as f:
        for item in listToSave:
            f.write("%s\n" % item)
        print("saving done")


def loadModel(model):
    if os.path.exists('./best_checkpoint.model'):
        checkpoint = torch.load('best_checkpoint.model')
        model.load_state_dict(checkpoint)
    return model


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def convertCat(cat):
    conversion = {
        0: 'chl_1', 1: 'chl_2', 2: 'chl_3', 3: 'chl_4', 4: 'chl_8', 5: 'chl_multi', 6: 'debr', 7: 'sp'
    }
    cat = conversion[cat]
    return cat


def trainModel(model, num_epochs, train_count, train, val, val_count, startTime, loss):
    trainlog = []
    vallog = []
    trainloss = []
    valloss = []
    for epoch in range(num_epochs):
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0

        val_acc = 0.0
        val_loss = 0.0

        best_acc = 0.0

        for i, (images, labels) in enumerate(train):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_accuracy += int(torch.sum(prediction == labels.data))

        # print(train_accuracy, " ", train_count)
        train_accuracy = train_accuracy / train_count
        train_loss = train_loss / train_count
        trainlog.append(train_accuracy)
        trainloss.append(train_loss.item())

        for i, (images, labels) in enumerate(val):
            optimizer.zero_grad()
            outputs = model(images)

            val_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            val_acc += int(torch.sum(prediction == labels.data))

        # print(train_accuracy, " ", train_count)
        val_acc = val_acc / val_count
        val_loss = val_loss / val_count
        vallog.append(val_acc)
        valloss.append(val_loss.item())
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_checkpoint.model')

        print(
            'Training till this point took ' + str(int(time.time() - startTime)) + ' seconds Epoch: ' + str(epoch)
            + ' Train Loss: ' + str(train_loss.item()) +" Val_loss: "+str(val_loss.item()) +' Train Accuracy: ' + str(int(train_accuracy*100)) +
            "% Validation Accuracy: " + str(int(val_acc*100))+"%")
    plt.plot(trainlog)
    plt.plot(vallog)
    plt.show()

    plt.plot(trainloss)
    plt.plot(valloss)
    plt.show()


def predict(model, testloader, test_path):
    model.eval()
    result = []
    for i, (images, labels) in enumerate(testloader):
        # Old version
        outputs = model(images)
        prediction = outputs.data.cpu().numpy().argmax()
        result.append(prediction)
    saveList(result)
    path = test_path + "/TEST"
    filelist = os.listdir(path)
    textfile = []
    textfile.append("Id,Category")
    for i in range(len(result)):
        line = str(filelist[i].split("_")[0]) + ", " + str(convertCat(result[i]))
        textfile.append(line)
    print(textfile)
    saveList(textfile)


def trainValSplit(input, split=0.1):
    original = len(input.dataset)

    vallen = int(math.floor(original * split))
    trainlen = int(original - vallen)
    print("Origianal length ", len(trainLoader.dataset), " train: ", trainlen, " Vallen: ", vallen)

    trn, val = random_split(input.dataset, [trainlen, vallen],
                            generator=torch.Generator().manual_seed(1))
    trn = DataLoader(trn)
    val = DataLoader(val)
    return trn, val, trainlen, vallen


if __name__ == "__main__":
    train_path = "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN"
    test_path = "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TEST_BMP"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer, transformer2 = preprocess()
    trainLoader, testLoader = dataLoader(train_path, test_path, transformer, transformer2)
    categories = loadCategories(train_path)

    train, val, train_count, val_count = trainValSplit(trainLoader)
    # weightlist = [1, 10, 10, 10, 10, 10, 10, 10]
    weightlist = [1, 6.6, 16.21, 9.216, 32.44, 42.685, 4.18, 4.53]
    loss_function = nn.CrossEntropyLoss(torch.FloatTensor(weightlist))
    num_epochs = 15
    # train_count = len(glob.glob(train_path + '/**/*.BMP')) * 2
    # test_count = len(glob.glob(test_path + '/**/*.BMP')) * 2

    model = ConvNet(num_classes=len(categories)).to(device)
    # model = loadModel(model)
    optimizer = Adam(model.parameters(), lr=0.005)
    # print("Number of training datapoints: ", train_count,
    # "\nNumber of testing datapoints: ", test_count)
    best_accuracy = 0.0
    startTime = time.time()
    print("Starting training phase")
    # Train model
    trainModel(model, num_epochs, train_count, trainLoader, val, val_count, startTime, loss_function)

    # Evaluation on testing dataset
    predict(model, testLoader, test_path)

    save = False
    if True:
        torch.save(model.state_dict(), 'best_checkpoint.model')
