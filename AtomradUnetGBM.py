from datetime import datetime
import torch
from torch.utils.data import random_split, DataLoader
import monai
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import pathlib

plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()


#Input data set

preprocess_val = tio.Compose([
    tio.RescaleIntensity((-1, 1)),
    tio.CropOrPad((240,240,155)), 
    tio.EnsureShapeMultiple(8),
    tio.OneHot(),
    ])

new_transform = tio.CropOrPad((240,240,155))

one_hot_transform = tio.OneHot()

print(f'working dir: {os.getcwd()}')
print(f'listdir: {os.listdir()}')
brats_root = pathlib.PosixPath(pathlib.Path.cwd(),"input")
print(f'brats_root: {brats_root}')
print(f'brats_root_list_dir: {os.listdir(brats_root)}')
brats_folder = pathlib.PosixPath(brats_root)
print(f'brats_folder: {brats_folder}')


patient_flair_list_brats = [pathlib.PosixPath(brats_folder, x) for x in os.listdir(brats_folder) if x[-12:] == 'flair.nii.gz']
patient_t1_list_brats = [pathlib.PosixPath(brats_folder, x) for x in os.listdir(brats_folder) if x [-9:] == 't1.nii.gz']
patient_t1ce_list_brats = [pathlib.PosixPath(brats_folder, x)  for x in os.listdir(brats_folder) if x [-11:] == 't1ce.nii.gz']
print(f'flair list: {patient_flair_list_brats}')
print(f't1 list: {patient_t1_list_brats}')
print(f't1ce list: {patient_t1ce_list_brats}')


brats_subjects = []
for a, b, c in zip(
    patient_flair_list_brats,
    patient_t1_list_brats,
    patient_t1ce_list_brats):

    subject = tio.Subject(
        channel_flair = tio.ScalarImage(a),
        channel_t1 = tio.ScalarImage(b),
        channel_t1ce = tio.ScalarImage(c),
        name = str(a)
    )
    brats_subjects.append(subject)

brats_dataset = tio.SubjectsDataset(brats_subjects, transform = preprocess_val)
brats_data_loader = DataLoader(brats_dataset, batch_size = 1)
print(f'brats dataset: {brats_dataset}')


train_subjects = brats_subjects
test_subjects = brats_subjects

print(f'train_subjects: {train_subjects}')
print(f'test_subjects: {test_subjects}')

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

    #can comment this out and CropOrPad for faster setup
    def get_max_shape(self, train_subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(train_subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad((240,240,160)), 
            tio.EnsureShapeMultiple(8),
            tio.OneHot(), 
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
        self.val_set = tio.SubjectsDataset(train_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size)  

new_data_module = DataModule(train_subjects=train_subjects, test_subjects=test_subjects, batch_size=1, train_val_ratio=0.8)
new_data_module.setup()

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
        return batch['channel_flair'][tio.DATA], batch['channel_t1'][tio.DATA], batch['channel_t1ce'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_batch(self, batch):
        e, f, g, y = self.prepare_batch(batch)
        batch_channel_tuple = (e, f, g)
        all_images = torch.cat(batch_channel_tuple, dim=1)
        y_hat = self.net(all_images)
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

unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=3,
    out_channels=5,
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
    gpus=None,
    precision= 16,
    callbacks=[early_stopping],
)
trainer.logger._default_hp_metric = False

model_file = pathlib.PosixPath(os.getcwd(),'Unet_Model_Multichannel_Statedict.pth')
print(os.getcwd())
print('model file')
print(model_file)
# torch.save(model, model_file)
model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
model.eval()


with torch.no_grad():
    for x in brats_data_loader:
        #Concatenate Channels
        flair_con = x['channel_flair'][tio.DATA]
        t1_con = x['channel_t1'][tio.DATA]
        t1ce_con = x['channel_t1ce'][tio.DATA]
        brats_batch_channel_tuple = (flair_con, t1_con, t1ce_con)
        all_images_brats = torch.cat(brats_batch_channel_tuple, dim=1).to(model.device)

        #Create predictions and add predictions to subjects
        preds = model.net(all_images_brats).argmax(dim=1, keepdim=True).cpu()
        batch_subject_brat = tio.utils.get_subjects_from_batch(x)
        tio.utils.add_images_from_batch(batch_subject_brat, preds, tio.LabelMap)
        # print(batch_subject_brat[0]['prediction'].plot())
        new_name = x['name'][0][-18:-13]
        print(new_name)
        transformed_batch_subject_brat = new_transform(batch_subject_brat[0]['prediction'])
        posix_path = pathlib.PosixPath(pathlib.Path.cwd(),"output")
        transformed_batch_subject_brat.save(pathlib.PosixPath(posix_path, f'{new_name}.nii.gz'))