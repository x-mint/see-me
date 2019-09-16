import sys
package_path = './input/unetmodelscript'
sys.path.append(package_path)

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose, Resize)
from albumentations.torch import ToTensor
from model import Unet
import matplotlib.pyplot as plt

WIDTH=224
HEIGHT=224
basePath = "/home/melodi_caliskan/SeeMe/"


class TestDataset(Dataset):
    def __init__(self, image_name):
        self.image_name = image_name
        self.num_samples = 1
        self.root = "/home/melodi_caliskan/SeeMe/server/images/uploads/"

        self.transform = Compose(
            [
                Resize(WIDTH, HEIGHT), Normalize(), ToTensor()
            ]
        )

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_name)
        img = cv2.imread(image_path)
        processed_image = self.transform(image=img)["image"]
        return self.image_name, processed_image

    def __len__(self):
        return self.num_samples


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((WIDTH, HEIGHT), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def rle2mask(rle, imgshape):
    w, h = imgshape[0], imgshape[1]
    mask = np.zeros(w * h).astype(np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts, lengths = array[0::2], array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]
    return np.flipud(np.rot90(mask.reshape(h, w), k=1))


def getModel():
    # Initialize mode and load trained weights
    model_path = os.path.join(basePath, "input/unetstartermodelfile/model_neu.pth")
    model = Unet("resnet50", classes=6)
    model.to(torch.device("cpu"))
    model.eval()
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    return model


def getPredictions(testset, model):
    # start prediction
    best_threshold = 0.18
    predictions = []
    for i, batch in enumerate(testset):
        fnames, images = batch
        device = torch.device("cpu")
        batch_preds = torch.sigmoid(model(images.to(device)))
        batch_preds = batch_preds.detach().cpu().numpy()
        for fname, preds in zip(fnames, batch_preds):
            for cls, pred in enumerate(preds):
                pred, num = post_process(pred, best_threshold, min_size)
                rle = mask2rle(pred)
                name = fname + "_" + str(cls + 1)
                predictions.append([name, rle])
    return predictions


def getDefectList(df_preds):
    defectedClasses = []
    labels = df_preds.iloc[0:6, 1]

    if pd.isnull(labels).all():
        print("No defect!\n")
    if pd.notnull(labels)[0]:
        defectedClasses.append(0)
        print("Defect found! Class: rolled-in_scale\n")
    if pd.notnull(labels)[1]:
        defectedClasses.append(1)
        print("Defect found! Class: patches\n")
    if pd.notnull(labels)[2]:
        defectedClasses.append(2)
        print("Defect found! Class: crazing\n")
    if pd.notnull(labels)[3]:
        defectedClasses.append(3)
        print("Defect found! Class: pitted_surface\n")
    if pd.notnull(labels)[4]:
        defectedClasses.append(4)
        print("Defect found! Class: inclusion\n")
    if pd.notnull(labels)[5]:
        defectedClasses.append(5)
        print("Defect found! Class: scratches\n")
    return defectedClasses


def getProcessedImage(test_data_folder, fname, df_preds, iloc_i):
    img = cv2.imread(os.path.join(test_data_folder, fname))
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    mask = rle2mask(df_preds['EncodedPixels'].iloc[iloc_i], img.shape)
    return img, mask


def plot_image(test_data_folder, fname, mask, text):
    img = cv2.imread(os.path.join(test_data_folder, fname))
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    mask2 = cv2.resize(mask, (w, h))
    contours, _ = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (249, 50, 12), 2)
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (114, 0, 218), 1)

    plt.title(fname)
    plt.imshow(img)
    plt.savefig(os.path.join(basePath, 'server/images/model_outputs/' + fname))
    plt.show()


def main(testset, test_data_folder, fname):
    # get model
    model = getModel()
    # get predictions
    predictions = getPredictions(testset, model)
    # save predictions to submission_tmp.csv
    df_preds = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df_preds.to_csv(os.path.join(basePath, "input/submission_tmp.csv"), index=0)
    df_preds = pd.read_csv(os.path.join(basePath, "input/submission_tmp.csv"))

    class_dict = {0: 'Rolled-in Scale', 1: 'Patches', 2: 'Crazing', 3: 'Pitted Surface', 4: 'Inclusion', 5: 'Scratches'}
    defectedClasses = getDefectList(df_preds)
    classes_images = []
    for class_ in defectedClasses:
        img, mask = getProcessedImage(test_data_folder, fname, df_preds, class_)
        classes_images.append([class_, img])
        plot_image(test_data_folder, fname, mask, class_dict.get(class_))

    return classes_images


if __name__ == '__main__':
    test_data_folder = os.path.join(basePath, "server/images/uploads/")

    # initialize test dataloader
    num_workers = 2
    min_size = 3500
    fname = sys.argv[1]

    testset = DataLoader(
        TestDataset(fname),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    main(testset, test_data_folder, fname)

