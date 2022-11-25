import csv
import os
import numpy as np
import time
import torch
import glob
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib


class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()
        # input shape:
        # (256,3,128,128)
        # output:
        # (width-kernelsize+2*padding)/stride+1
        # (128-3+2)+1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Change of shape (256,12,128,128)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Change of shape: (256,12,64,64)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Change of shape: (256,20,64,64)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Change of shape: (256,32,64,64)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.LeakyReLU()

        self.fc = nn.Linear(in_features=32 * 64 * 64, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 64 * 64)

        output = self.fc(output)

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


def trainModel(model, num_epochs, train_count, startTime):
    for epoch in range(num_epochs):

        model.train()
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(trainLoader):
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
        print(
            'Training till this point took ' + str(int(time.time() - startTime)) + ' seconds Epoch: ' + str(
                epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(train_accuracy) + "\n")


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


if __name__ == "__main__":
    train_path = "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN"
    test_path = "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TEST_BMP"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer, transformer2 = preprocess()
    trainLoader, testLoader = dataLoader(train_path, test_path, transformer, transformer2)
    categories = loadCategories(train_path)

    loss_function = nn.CrossEntropyLoss()
    num_epochs = 10
    train_count = len(glob.glob(train_path + '/**/*.BMP')) * 2
    test_count = len(glob.glob(test_path + '/**/*.BMP')) * 2
    model = ConvNet(num_classes=len(categories)).to(device)
    model = loadModel(model)
    optimizer = Adam(model.parameters(), lr=0.0001)
    # print("Number of training datapoints: ", train_count,
    # "\nNumber of testing datapoints: ", test_count)

    best_accuracy = 0.0
    startTime = time.time()
    print("Starting training phase")
    # Train model
    trainModel(model, num_epochs, train_count, startTime)

    # Evaluation on testing dataset
    predict(model, testLoader, test_path)

    save = True
    if True:
        torch.save(model.state_dict(), 'best_checkpoint.model')
