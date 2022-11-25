import cv2 as cv
import numpy as np
import os

tmp_all_images = []
all_images = []
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/sp"):
    with open(os.path.join(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/sp",
            filename), 'r') as f:
        image = cv.imread(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/sp/" + filename)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "sp"])
        # cv.imshow("sp", image)
for filename in os.listdir(
        "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/debr"):
    with open(os.path.join(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/debr",
            filename), 'r') as f:
        image = cv.imread(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/debr/" + filename)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "debr"])
for filename in os.listdir(
        "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_multi"):
    with open(os.path.join(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_multi",
            filename), 'r') as f:
        image = cv.imread(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_multi/" + filename)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_multi"])
for filename in os.listdir(
        "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_8"):
    with open(os.path.join(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_8",
            filename), 'r') as f:
        image = cv.imread(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_8/" + filename)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_8"])
for filename in os.listdir(
        "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_4"):
    with open(os.path.join(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_4",
            filename), 'r') as f:
        image = cv.imread(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_4/" + filename)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_4"])
for filename in os.listdir(
        "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_3"):
    with open(os.path.join(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_3",
            filename), 'r') as f:
        image = cv.imread(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_3/" + filename)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_3"])
for filename in os.listdir(
        "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_2"):
    with open(os.path.join(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_2",
            filename), 'r') as f:
        image = cv.imread(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_2/" + filename)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_2"])
for filename in os.listdir(
        "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_1"):
    with open(os.path.join(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_1",
            filename), 'r') as f:
        image = cv.imread(
            "c:/Users/kori/Documents/Egyetem/neurhalok/competition/ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_1/" + filename)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_1"])

for i in range(0, len(tmp_all_images)):
    if i % 2 == 0:
        # all_images.append([np.concatenate(tmp_all_images[i][0], tmp_all_images[i+1][0]),tmp_all_images[i][0]])
        #  all_images.append([tmp_all_images[i][0]+tmp_all_images[i+1][0], tmp_all_images[i][1]])  ez is működik
        # all_images.append([tmp_all_images[i][0], tmp_all_images[i + 1][0], tmp_all_images[i][1]])
        all_images.append([np.concatenate((tmp_all_images[i][0], tmp_all_images[i + 1][0],np.zeros((128,128,1))),axis=2),tmp_all_images[i][1]])
    else:
        continue

print(all_images)

