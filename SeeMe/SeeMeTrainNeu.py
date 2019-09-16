import os
import time
import warnings
import random
import argparse
import torch
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from albumentations import (HorizontalFlip, Normalize, Compose)
from albumentations.torch import ToTensor

import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True  # to get deterministic behaviour

DEFECTS = {'rolled-in_scale': 0, 'patches': 1, 'crazing': 2, 'pitted_surface': 3, 'inclusion': 4, 'scratches': 5}
WIDTH = 224
HEIGHT = 224
NEU_class = 6


class SteelDataset(Dataset):
    def __init__(self, image_names, annotations, image_folder, phase):
        self.image_names = image_names
        self.annotations = annotations
        self.image_folder = image_folder
        self.phase = phase
        self.transforms = get_transforms(phase)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        annotation = self.annotations.loc[self.annotations['Image'] == image_name]
        mask = make_mask(annotation)
        image_path = os.path.join(self.image_folder, image_name + '.jpg')
        img = cv2.imread(image_path)
        img = cv2.resize(img, (WIDTH, HEIGHT
                               ), interpolation=cv2.INTER_AREA)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)
        return img, mask

    def __len__(self):
        return len(self.image_names)


class Trainer(object):
    def __init__(self, model, train_settings):
        self.train_path = train_settings['train_path']
        self.image_folder = train_settings['image_folder']
        self.best_model_path = train_settings['best_model_path']
        self.num_workers = train_settings['num_workers']
        self.batch_size = train_settings['batch_size']
        self.accumulation_steps = train_settings['acc_step'] // self.batch_size['train']
        self.lr = train_settings['learning_rate']
        self.num_epochs = train_settings['epochs']
        self.best_loss = float("inf")
        self.phases = ["train", "val"]

        self.device = torch.device("cuda:0")
        self.net = model.to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=train_settings['patience'],
                                           verbose=True)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cudnn.benchmark = True

        self.dataloaders = {
            phase: provider(
                train_path=self.train_path,
                image_folder=self.image_folder,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }

        self.losses = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        print("Phase: %s | Epoch: %s | Start Time: %s" % (phase, epoch, time.strftime("%H:%M:%S")))
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        print("Loss: %0.5f" % (epoch_loss))
        self.losses[phase].append(epoch_loss)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("Saving best model...")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, self.best_model_path)
            print()


def get_transforms(phase):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(),
            ]
        )
    list_transforms.extend(
        [
            Normalize(),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def provider(
        train_path,
        image_folder,
        phase,
        batch_size=8,
        num_workers=4,
):
    df = pd.read_csv(train_path, sep=';')
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["name"])
    df = train_df if phase == "train" else val_df
    image_names = list(set(df['Image']))
    image_dataset = SteelDataset(image_names, df, image_folder, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


def make_mask(annotation):
    masks = np.zeros((HEIGHT, WIDTH, NEU_class), dtype=np.float32)
    mask = np.zeros([HEIGHT, WIDTH], dtype=np.uint8)
    for idx, row in annotation.iterrows():
        defect_name = row['name']
        mask[row['xmin']:row['xmax'], row['ymin']:row['ymax']] = 1
    try:
        masks[:, :, DEFECTS[defect_name]] = mask
    except:
        print()
    return masks


def plot(scores, model_loss_path):
    plt.figure()
    plt.plot(range(len(scores["train"])), scores["train"], label='train BCE loss')
    plt.plot(range(len(scores["train"])), scores["val"], label='val BCE loss')
    plt.title('BCE loss plot')
    plt.xlabel('Epoch')
    plt.ylabel('BCE loss')
    plt.legend()
    plt.savefig(model_loss_path)
    plt.close()


if __name__ == '__main__':

    # base_path = '/home/melodi_caliskan/SeeMe/'

    parser = argparse.ArgumentParser(prog='see-me')
    parser.add_argument('-rp', '--rootpath')
    args = parser.parse_args()
    root_path = args.rootpath
    if not root_path:
        root_path = os.getcwd()

    train_path = os.path.join(root_path, 'input', 'annotations.csv')
    image_folder = os.path.join(root_path, 'input', 'images')
    model_output_folder = os.path.join(root_path, 'pretrainedmodels')
    best_model_path = os.path.join(model_output_folder, 'model_neu.pth')
    model_loss_path = os.path.join(model_output_folder, 'train_val_loss.png')

    train_settings = {'num_workers': 6, 'batch_size': {'train': 4, 'val': 4},
                      'acc_step': 32, 'learning_rate': 5e-4, 'epochs': 20, 'patience': 3,
                      'train_path': train_path,
                      'image_folder': image_folder,
                      'best_model_path': best_model_path}

    model = smp.Unet("resnet50", encoder_weights="imagenet", classes=6, activation=None)
    model_trainer = Trainer(model, train_settings)
    model_trainer.start()

    # PLOT TRAINING
    losses = model_trainer.losses
    plot(losses, model_loss_path)
