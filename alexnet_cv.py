import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import transforms, models, datasets
import time
import copy
import matplotlib.pyplot as plt


def plot_epoch(epoch_list, train_acc, val_acc, train_loss, val_loss, k=1):
    fig_epoch = plt.figure(k)

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


# Data augmentation and normalization for training
# Just normalization for validation
crop_list = [transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]), transforms.RandomResizedCrop(224)]

data_transforms = {
    'train': transforms.Compose([
         transforms.RandomChoice(crop_list),
        transforms.RandomHorizontalFlip(),
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

def datasets_K_fold_CV(dataset, n_fold=5):
    n_images = len(dataset)

    lengths_seq = (n_fold-1) * [n_images//n_fold]
    lengths_seq += [n_images - sum(lengths_seq)]
    list_subdataset = torch.utils.data.random_split(dataset, lengths_seq)

    list_trainset = []
    list_testset = []
    for k in range(n_fold):
        list_testset.append(list_subdataset[k])
        train_k_conc = torch.utils.data.ConcatDataset(list_subdataset[:k] + list_subdataset[k + 1:])
        list_trainset.append(train_k_conc)

    return list_trainset, list_testset


#os.system("pause")
dataloaders = torch.utils.data.DataLoader(image_datasets,
                                          batch_size=50, shuffle=True, num_workers=4)
print(len(dataloaders), type(dataloaders), dataloaders)

dataset_sizes = len(image_datasets)
class_names = image_datasets.classes
nb_classes = len(class_names)

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


def confusion_matrix(model_ft, n_classes):

    conf_matrix = torch.zeros(n_classes, n_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    # percentage properly classified, from a class
    accuracy_from_class = conf_matrix.diag() / conf_matrix.sum(1)

    # percentage classified in a class which are properly classified
    accuracy_to_class = conf_matrix.diag() / conf_matrix.sum(2)

    return conf_matrix, accuracy_from_class, accuracy_to_class


def train_AlexNet(dataset, deep_model=models.alexnet, pretrained=True):
    model_ft = deep_model(pretrained=pretrained)

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
    # Decay LR by a factor of "gamma" every "step_size" epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=31)

    return model_ft


if __name__ == '__main__':
    torch.cuda.empty_cache()

    path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"
    image_datasets = datasets.ImageFolder(path)

    model_ft = train_AlexNet(image_datasets, pretrained=True)

    plt.show()