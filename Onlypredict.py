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
from Prototype import ConvNet, preprocess, dataLoader, loadCategories, loadModel, predict

import torchvision
import pathlib
import math

if __name__ == "__main__":
    train_path = "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN"
    test_path = "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TEST_BMP"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer, transformer2 = preprocess()
    trainLoader, testLoader = dataLoader(train_path, test_path, transformer, transformer2)
    categories = loadCategories(train_path)
    weightlist = [1, 6.6, 16.21, 9.216, 32.44, 42.685, 4.18, 4.53]
    loss_function = nn.CrossEntropyLoss(torch.FloatTensor(weightlist))
    model = ConvNet(num_classes=len(categories)).to(device)
    optimizer = Adam(model.parameters(), lr=0.005)
    model = loadModel(model)
    predict(model, testLoader, test_path)


