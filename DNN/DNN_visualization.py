import matplotlib.pyplot as plt
import torch
import numpy as np

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_epoch(epoch_list, train_acc, val_acc, train_loss, val_loss, figure=0):
    """ Plot the accuracy and the loss for the training and the validation set with respect to the epoch """

    # create the figure
    fig_epoch = plt.figure(figure)

    # plot the accuracy on both the training and the validation set
    ax1 = fig_epoch.add_subplot(1, 2, 1)
    plt.plot(epoch_list, train_acc, 'b', label='train')
    plt.plot(epoch_list, val_acc, 'r', label='val')
    ax1.title.set_text('Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    # plot the loss on both the training and the validation set
    ax2 = fig_epoch.add_subplot(1, 2, 2)
    plt.plot(epoch_list, train_loss, 'b', label='train')
    plt.plot(epoch_list, val_loss, 'r', label='val')
    ax2.title.set_text('Loss')
    plt.legend()

    return fig_epoch


def confusion_matrix(model_ft, n_classe, dataloaders, device=default_device):
    """ compute the confusion matrix of a pytorch model"""

    # initialize the confusion matrix
    conf_matrix = torch.zeros(n_classe, n_classe)

    # enable the computation without training
    with torch.no_grad():
        for i, (inputs, label) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(label.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    # percentage properly classified, from a class
    accuracy_from_class = conf_matrix.diag() / conf_matrix.sum(dim=1)

    # percentage classified in a class which are properly classified
    # TODO: change the 0 in the sum to sum on line instead of column
    accuracy_to_class = conf_matrix.diag() / conf_matrix.sum(dim=0)

    return np.array(conf_matrix), accuracy_from_class, accuracy_to_class
