import cv2
import cv2 as cv
import numpy as np
import os
import torchvision.transforms as transforms

def rotate(classes):
    for filename in os.listdir(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN_merged/chl_multi"):
        with open(os.path.join("./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN_merged/chl_multi", filename), 'r') as f:
            image = cv.imread("./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN_merged/chl_multi/" + filename)
            r = cv2.imwrite(
                "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN_merged/"+classes+"/rot1_" + filename,
                cv.rotate(image, cv2.ROTATE_90_CLOCKWISE))
            r = cv2.imwrite(
                "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN_merged/"+classes+"/rot2_" + filename,
                cv.rotate(image, cv2.ROTATE_180))
            r = cv2.imwrite(
                "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN_merged/"+classes+"/rot3_" + filename,
                cv.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

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
TEST_tmp = []
TEST_all = []
for filename in os.listdir(
        "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TEST/TEST"):
    with open(os.path.join(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TEST/TEST",
            filename), 'r') as f:
        image = cv.imread(
            "./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TEST/TEST/" + filename,
            cv2.IMREAD_GRAYSCALE)
        # image_2 = cv.imread()
        TEST_tmp.append([image, filename])
for i in range(0, len(TEST_tmp)):
    if i % 2 == 0:
        TEST_all.append([np.concatenate((TEST_tmp[i][0].reshape((128, 128, 1)),
                                           TEST_tmp[i + 1][0].reshape((128, 128, 1)), np.zeros((128, 128, 1))),
                                          axis=2) / 255, TEST_tmp[i][1].split('_')[0]])
    else:
        continue
# for i in range(0, len(TEST_all)):
#     actual_image_name = TEST_all[i][1]
#     r = cv2.imwrite("./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TEST_merged/" + actual_image_name + '.png', TEST_all[i][0] * 255)


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
# for i in range(0, len(all_images)):
#     actual_image_name = all_images[i][2]
#     r = cv2.imwrite("./ppke-itk-neural-networks-2022-challenge/db_chlorella_renamed_TRAIN_merged/" + all_images[i][1] + "/" + actual_image_name + '.png', all_images[i][0] * 255)
#
# print(all_images)

# rotate("chl_multi")
rotate("chl_8")
