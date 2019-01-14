import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms, models, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import copy

torch.cuda.empty_cache()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Data augmentation and normalization for training
# Just normalization for validation
crop_list = [transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])] + 2 * [transforms.RandomResizedCrop(224)]

data_transforms = {
    'train': transforms.Compose([
         transforms.RandomChoice(crop_list),
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset_alexnet/"

image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                          data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=50,
                                             shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
nb_classes = len(class_names)
#print('nb_classes =', nb_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch_list = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_list.append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
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
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=50):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(1)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//10, 10, images_so_far)
                ax.axis('off')
                if preds[j] == labels[j].data:
                    ax.set_title('{}'.format(class_names[preds[j]]))
                else:
                    ax.set_title('{}'.format(class_names[preds[j]]), color='red')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    model_ft = models.alexnet(pretrained=True)
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, nb_classes)

    model_ft.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, nb_classes),
    )

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=31)

    visualize_model(model_ft)

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    confusion_matrix_percent = confusion_matrix.diag() / confusion_matrix.sum(1)

    print(class_names)
    print(confusion_matrix, confusion_matrix_percent)

    fig_epoch = plt.figure(2)

    ax1 = fig_epoch.add_subplot(121)
    plt.plot(epoch_list, train_acc, 'b', label='train')
    plt.plot(epoch_list, val_acc, 'r', label='val')
    ax1.title.set_text('Accuracy')
    plt.ylim(0,1)
    plt.legend()

    ax2 = fig_epoch.add_subplot(122)
    plt.plot(epoch_list, train_loss, 'b', label='train')
    plt.plot(epoch_list, val_loss, 'r', label='val')
    ax2.title.set_text('Loss')
    plt.legend()

    plt.show()