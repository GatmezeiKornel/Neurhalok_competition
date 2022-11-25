import cv2
import cv2 as cv
import numpy as np
import os

tmp_all_images = []
all_images = []
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/sp"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/sp",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/sp/" + filename, cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "sp", filename])
        # cv.imshow("sp", image)
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/debr"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/debr",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/debr/" + filename,
            cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "debr", filename])
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_multi"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_multi",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_multi/" + filename,
            cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_multi", filename])
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_8"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_8",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_8/" + filename,
            cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_8", filename])
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_4"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_4",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_4/" + filename,
            cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_4", filename])
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_3"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_3",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_3/" + filename,
            cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_3", filename])
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_2"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_2",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_2/" + filename,
            cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_2", filename])
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_1"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_1",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN/chl_1/" + filename,
            cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        tmp_all_images.append([image, "chl_1", filename])

for i in range(0, len(tmp_all_images)):
    if i % 2 == 0:
        # all_images.append([np.concatenate(tmp_all_images[i][0], tmp_all_images[i+1][0]),tmp_all_images[i][0]])
        #  all_images.append([tmp_all_images[i][0]+tmp_all_images[i+1][0], tmp_all_images[i][1]])  ez is működik
        # all_images.append([tmp_all_images[i][0], tmp_all_images[i + 1][0], tmp_all_images[i][1]])
        all_images.append([np.concatenate((tmp_all_images[i][0].reshape((128, 128, 1)),
                                           tmp_all_images[i + 1][0].reshape((128, 128, 1)), np.zeros((128, 128, 1))),
                                          axis=2) / 255, tmp_all_images[i][1], tmp_all_images[i][2].split('_')[0]])
    else:
        continue
for i in range(0, len(all_images), 2):
    actual_image_name = all_images[i][2]
    r = cv2.imwrite("./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN_merged/" + all_images[i][1] + "/" + actual_image_name + '.png', all_images[i][0] * 255)

print(all_images)
