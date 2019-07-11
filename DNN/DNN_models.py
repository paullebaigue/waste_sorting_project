import numpy as np
import time
import copy
import torch
import torch.nn as nn
from torchvision import models, transforms
from DNN.DNN_visualization import plot_epoch, confusion_matrix

defaut_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_transformation(mean_RGB, std_RGB):
    """ Define the transformations to apply to the immages before feeding them into the NNet """

    # some random resize choices for the training set
    crop_choice = transforms.RandomChoice([
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)]),
        transforms.RandomResizedCrop(224)])

    # Resize to the right shape and color normalization
    # Train: some data augmentation to reduce over-fitting
    # Val: no data augmentation for the validation
    transformation = {
        'train':
            transforms.Compose([
                crop_choice,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean_RGB, std_RGB)]),
        'val':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean_RGB, std_RGB)])}

    return transformation


def compile_model(deep_model=models.alexnet, n_classes=6, device=defaut_device):
    """ Load the parameters of the pre-trained model and change its dense classifier"""

    # Load/Download the pre-trained model
    model = deep_model(pretrained=True)

    # use a new classifier, smaller for a smaller database
    model.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(64, n_classes))

    model = model.to(device)

    return model


def split_train_val(dataset, test_size=0.3, transform_function=None):
    """ Split a pytorch loaded database into training and validation database"""

    # length of the different datasets
    n_image = len(dataset)
    n_val = int(test_size * n_image)
    n_train = n_image - n_val

    # split the dataset in subdataset of size n_train and n_val
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, (n_train, n_val))

    # apply the images transformation and resizing
    if transform_function is not None:
        dataset_train = [(transform_function["train"](image), label) for (image, label) in dataset_train]
        dataset_val = [(transform_function["val"](image), label) for (image, label) in dataset_val]

    # make a single dictionary out of the 2 sub-datasets
    transformed_dataset = {'train': dataset_train, 'val': dataset_val}

    return transformed_dataset


def one_epoch(model, phase, dataloaders, optimizer, loss_function, device=defaut_device):
    """ compute one epoch and optimize the NNet if in training mode"""

    # initialization of loss and accuracy of the epoch
    current_loss = 0.0
    current_accuracy = 0.0

    # iteration over the dataset
    for inputs, labels in dataloaders[phase]:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_accuracy += torch.sum(preds == labels.data)

    return current_loss, current_accuracy


def train_model(model, n_epochs, dataset_size, dataloaders,
                optimizer, scheduler, loss_funtion,
                display_sample=False):
    """ Train a model, plot the loss and accuracy during the training and display the confusion matrix"""

    # time measurement
    since = time.time()

    # initialisation
    best_acc = 0.0
    best_loss = np.inf
    epoch_list = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # iteration over the epochs
    for current_epoch in range(n_epochs):
        epoch_list.append(current_epoch)
        print('Epoch {}/{}'.format(current_epoch + 1, n_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # compute the epoch
            current_loss, current_accuracy = one_epoch(model=model, phase=phase,
                                                       dataloaders=dataloaders,
                                                       optimizer=optimizer, loss_function=loss_funtion)

            # compute accuracy and loss of the epoch
            epoch_loss = current_loss / dataset_size[phase]
            epoch_acc = current_accuracy.double() / dataset_size[phase]

            # print the result at the end of every epoch
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # create the lists of loss an accuracy with respect to the epoch
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    plot_epoch(epoch_list, train_acc, val_acc, train_loss, val_loss, figure=0)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} \n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # compute the confusion matrix
    conf_matrix, accuracy_from_class, accuracy_to_class = confusion_matrix(model, n_classe=6, dataloaders=dataloaders)

    return model, best_acc, conf_matrix
