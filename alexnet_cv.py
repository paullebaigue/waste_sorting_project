import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import transforms, models, datasets
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_epoch(epoch_list, train_acc, val_acc, train_loss, val_loss, k=0):

    fig_epoch = plt.figure(k+1)

    ax1=fig_epoch.add_subplot(1, 2, 1)
    plt.plot(epoch_list, train_acc, 'b', label='train')
    plt.plot(epoch_list, val_acc, 'r', label='val')
    ax1.title.set_text('Accuracy')
    plt.ylim(0,1)
    plt.legend()

    ax2=fig_epoch.add_subplot(1, 2, 2)
    plt.plot(epoch_list, train_loss, 'b', label='train')
    plt.plot(epoch_list, val_loss, 'r', label='val')
    ax2.title.set_text('Loss')
    plt.legend()

# Data augmentation and normalization for training
# Just normalization for validation
crop = transforms.RandomChoice([
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]),
    transforms.RandomResizedCrop(224),
])


data_transforms = {
    'train': transforms.Compose([
        crop,
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
        transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def datasets_K_fold_CV(dataset, transform_function=None, n_fold=5):
    n_images = len(dataset)

    lengths_seq = (n_fold-1) * [n_images//n_fold]
    lengths_seq += [n_images - sum(lengths_seq)]
    list_subdataset = torch.utils.data.random_split(dataset, lengths_seq)

    list_dataset = []
    for k in range(n_fold):
        train_k_conc = torch.utils.data.ConcatDataset(list_subdataset[:k] + list_subdataset[k + 1:])

        if transform_function is not None:
            print("transformation of data in datasets_K_fold_CV")
            train_k = [(transform_function["train"](image[0]), image[1]) for image in train_k_conc]
            val_k = [(transform_function["val"](image[0]), image[1]) for image in list_subdataset[k]]

        else:
            train_k = train_k_conc
            val_k = list_subdataset[k]

        dataset_k = {'train': train_k, 'val': val_k}
        list_dataset.append(dataset_k)

    return list_dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model_cv(data_k, deep_model=models.alexnet, n_classe=6, pretrained=True, k=1, num_epochs=25):
    model = deep_model(pretrained=pretrained)

    model.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, n_classe),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
    # Decay LR by a factor of "gamma" every "step_size" epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    ##################################################################################

    since = time.time()

    dataloaders = {x: torch.utils.data.DataLoader(data_k[x], batch_size=3,
                                                  shuffle=True, num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(data_k[x]) for x in ['train', 'val']}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10
    epoch_list = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        epoch_list.append(epoch)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
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
                optimizer_ft.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

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
            if phase == 'val' and epoch_acc > best_acc and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    plot_epoch(epoch_list, train_acc, val_acc, train_loss, val_loss, k=k)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} \n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    conf_matrix, accuracy_from_class, accuracy_to_class = confusion_matrix(model, n_classe, dataloaders)

    return model, best_acc, conf_matrix


def confusion_matrix(model_ft, n_classe, dataloaders):

    conf_matrix = torch.zeros(n_classe, n_classe)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    # percentage properly classified, from a class
    accuracy_from_class = conf_matrix.diag() / conf_matrix.sum(dim=1)

    # percentage classified in a class which are properly classified
    # TODO: change the 0 in the sum to sum on line instead of column
    accuracy_to_class = conf_matrix.diag() / conf_matrix.sum(dim=0)

    return np.array(conf_matrix), accuracy_from_class, accuracy_to_class


def train_AlexNet_cv(dataset, n_epoch=10, n_classe=6, n_fold=5, deep_model=models.alexnet, pretrained=True):

    subdataset = datasets_K_fold_CV(dataset, n_fold=n_fold)
                                    #transform_function=data_transforms)

    list_acc = []
    list_mat = []
    for k in range(n_fold):
        print(20*".")
        print('Fold %i/%i' %(k+1, n_fold))
        print(20*".")
        model, val_acc, conf_matrix = train_model_cv(subdataset[k], k=k, num_epochs=n_epoch,
                                                     n_classe=n_classe, deep_model=deep_model)

        list_acc.append(val_acc)
        list_mat.append(conf_matrix)

    return list_acc, list_mat


if __name__ == '__main__':
    torch.cuda.empty_cache()
    N_FOLD = 5
    N_EPOCH = 2

    path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"
    # it seems that it is better with the same transformation for all than with a data augmentation on the training set
    image_datasets = datasets.ImageFolder(path,
                                          transform=data_transforms['val'])
    class_names = image_datasets.classes
    n_classe = len(class_names)

    list_acc, list_matrix = train_AlexNet_cv(image_datasets,
                                             n_classe=n_classe, n_epoch=N_EPOCH, n_fold=N_FOLD, pretrained=True,
                                             deep_model=models.resnet50)

    total_conf_mat = np.sum(list_matrix, axis=0).astype(int)
    normed_TCM = total_conf_mat/np.sum(total_conf_mat, axis=1)

    print("Cross Validation accuracy :", round(float(np.sum(list_acc)/N_FOLD), 4))
    print("Total confusion matrix : \n", total_conf_mat)

    plt.figure(0)
    plt.title("Total confusion matrix")
    plot_conf_matrix = sns.heatmap(normed_TCM, annot=total_conf_mat, fmt="d", cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Class')
    plt.show()
