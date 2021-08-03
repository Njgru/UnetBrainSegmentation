

from pathlib import Path
from datetime import datetime
import os

import torch
from torch.utils.data import random_split, DataLoader
import monai
import gdown
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()

# %% [markdown]
# This method uses Pylightning to create a 'data module' class which utilizes Torchio's subject/image classes to create the data sets, which can take multiple different file types including DICOM and nii.
# 
# Pylightning is then used to create the 'module' training loop class, which also loads the 3d Monai U-net class within it.


root = os.getcwd()
Task1Folder = f'{root}\\Task1Synapse'
os.listdir(Task1Folder)
print(Task1Folder)



patient_list = [Task1Folder + '\\' + x +f'\{x}_flair.nii.gz' for x in os.listdir(Task1Folder)]

label_list = [Task1Folder + '\\' + x +f'\{x}_seg.nii.gz' for x in os.listdir(Task1Folder)]

subjects = []
for patient_list, label_list in zip(patient_list, label_list):
    subject = tio.Subject(
        image = tio.ScalarImage(patient_list),
        label = tio.LabelMap(label_list)
    )
    subjects.append(subject)



dataset1 = tio.SubjectsDataset(subjects)
print(f'number of subjects: {len(dataset1)}')
print(f'subject 1: {dataset1[0]}')
print(f'subject 1 image data: {dataset1[0].image}')
print(f'subject 1 label data: {dataset1[0].label}')
print(dataset1[0].image.spatial_shape)
print(dataset1[0].spatial_shape)



print(dataset1[0].image)
print(dataset1[0].label)



train_subjects = subjects[0:1200]
test_subjects = subjects[1200:]



class DataModule(pl.LightningDataModule):
    def __init__(self, train_subjects, test_subjects, batch_size, train_val_ratio):
        super().__init__()
        self.subjects = train_subjects
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.test_subjects = test_subjects
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    
    def get_max_shape(self, train_subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(train_subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad((240,240,160)), 
            tio.EnsureShapeMultiple(8),  # for the U-Net
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])
        return augment    

    def setup(self):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])
    
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size)  



new_data_module = DataModule(train_subjects=train_subjects, test_subjects=test_subjects, batch_size=1, train_val_ratio=0.8)


new_data_module.setup()



print(f'Training Subjects: {len(new_data_module.train_set)}')
print(f'Validation Subjects: {len(new_data_module.val_set)}')
print(f'Test Subjects: {len(new_data_module.test_set)}')


print(new_data_module.train_set[0].image)
print(new_data_module.train_set[0].label)



class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss



num_epochs = 3

unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1, 
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
)

model = Model(
    net=unet,
    criterion=monai.losses.DiceCELoss(softmax=True),
    learning_rate=1e-2, 
    optimizer_class=torch.optim.AdamW,
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)
trainer = pl.Trainer(
    max_epochs = num_epochs,
    gpus=1,
    precision= 16, 
    callbacks=[early_stopping],
)
trainer.logger._default_hp_metric = False



start = datetime.now()
print('Training started at', start)
trainer.fit(model=model, datamodule=new_data_module) #new_data_module instead of data
print('Training duration:', datetime.now() - start)


