from DNN.DNN_models import train_model, split_train_val, compile_model, image_transformation
import torch
from torch.optim import lr_scheduler, SGD, Adam
from torchvision import models, datasets
import torch.nn as nn
import matplotlib.pyplot as plt

torch.cuda.current_device()

# path to the dataset
path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"

# normalisation of the RGB channels of the images considering the distribution of the dataset
mean_RGB, std_RGB = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# parameters for the training
EPOCH = 30
BATCH = 8
LR = 0.0001
GAMMA = 0.05
MOMENTUM = 0.9
STEP_SIZE = 10
TEST_SIZE = 0.3

# transformation to prevent over-fitting and to reshape the image to the right size
transformation = image_transformation(mean_RGB, std_RGB)

# load the dataset from its folder
image_datasets = datasets.ImageFolder(path)

# split train and validation and transform the data
transformed_dataset = split_train_val(image_datasets, test_size=TEST_SIZE,
                                      transform_function=transformation)

# load progressively the data with the CPU while the GPU is computing
# this aims at running the GPU full capacity and avoiding bottlenecks in the data pipeline
dataloaders = {phase: torch.utils.data.DataLoader(transformed_dataset[phase], batch_size=BATCH,
                                              shuffle=True, num_workers=0) for phase in ['train', 'val']}

# the number of images in each dataset: used to compute the loss and the accuracy
dataset_sizes = {phase: len(transformed_dataset[phase]) for phase in ['train', 'val']}

# load and enable the model
model = compile_model(deep_model=models.alexnet, n_classes=6)

# training optimization functions
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)  # momentum=MOMENTUM)
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# fit the model to the training data and evaluate the validation data
train_model(model, n_epochs=EPOCH, dataset_size=dataset_sizes, dataloaders=dataloaders,
            optimizer=optimizer, scheduler=scheduler, loss_funtion=loss_function,
            display_sample=True)

plt.show()
