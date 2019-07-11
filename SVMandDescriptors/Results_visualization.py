import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy.random as rd
import numpy as np
from sklearn import cluster
from sklearn.model_selection import train_test_split
from SVMandDescriptors.SVM_models import cross_validation, learn_SVM


def display_test_images(X_image_test, Y_test, Y_test_pred):
    """ plot a sample of images and their labels as titles
        if the label is white the image is well classified, if it is red it is a bad classification
    """

    fig = plt.figure()

    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    test_size = len(X_image_test)

    assert test_size > 40

    c = 0
    used_idx = []
    # display 40 images
    while c < 40:

        # select the images randomly
        idx_img = rd.randint(0, test_size)

        # ensure an image is displayed only once
        if idx_img not in used_idx:
            c += 1
            used_idx.append(idx_img)
            plt.subplot(5, 8, c, xticks=[], yticks=[])
            plt.imshow(X_image_test[idx_img], cmap='gray')

            # title format: prediction / real
            title = str(classes[Y_test_pred[idx_img] - 1] + ' / ' + str(classes[Y_test[idx_img] - 1]))

            # select the title color: well predicted = white, wrong = red
            if Y_test_pred[idx_img] != Y_test[idx_img]:
                plt.text(0.1, 0.1, title,
                         fontsize=6, bbox=dict(facecolor='red', alpha=1))
            else:
                plt.text(0.1, 0.1, title,
                         fontsize=6, bbox=dict(facecolor='white', alpha=1))
    return fig


def choose_score_function(X, y, test_size=0.2,
                          cross_validate=False, n_fold=10, kernel='linear'):
    """ select the score function if cross validation is used or not
        cross validation takes more time to compute but is more reliable
    """

    if cross_validate:
        # returns a function giving the score in CV case
        def score_function(C, gamma):
            loss, score, _, _ = cross_validation(X, y,
                                                 n_fold=n_fold,
                                                 kernel=kernel,
                                                 C=C, gamma=gamma,
                                                 print_result=False)
            return loss, score

    else:
        indices = range(len(X))
        X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(X, y, indices,
                                                                                    test_size=test_size, shuffle=True)
        # returns a function giving the score in a case of a single split train/test
        def score_function(C, gamma):
            _, loss, score, _ = learn_SVM(X_train, y_train, X_valid, y_valid,
                                          kernel=kernel,
                                          C=C, gamma=gamma,
                                          print_result=False)
            return loss, score

    return score_function


def plot_colormap(x_list, y_list, score_matrix, subplot=(1,1,1), score_max=1, score_min=1/3, title=''):
    """ plot a colormap from a matrix of values """

    plt.subplot(*subplot)

    # plot the color map and the legend bar
    plt.pcolor(x_list, y_list, np.transpose(score_matrix), cmap='RdBu_r', norm=colors.LogNorm(),
               vmax=score_max, vmin=score_min)
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar()
    plt.title(title)

    return


def SVM_colormap_C_gamma(X, y, test_size=0.2,
                         cross_validate=False, n_fold=10, kernel='linear',
                         C_power_range=range(-5,6,1), gamma_power_range=None,
                         figure=0):
    """ compute a matrix of score from two lists of parameters, and plot the associated color map
        can be used with or without cross validation
    """

    best_loss, best_score, best_C, best_gamma = np.inf, 0, None, None

    # define the function used to evaluate the score
    score_function = choose_score_function(X, y, cross_validate=cross_validate,
                                           test_size=test_size, n_fold=n_fold, kernel=kernel)

    loss_matrix, score_matrix = [], []
    C_list = 10**np.array(C_power_range, dtype=float)
    gamma_list = 10**np.array(gamma_power_range, dtype=float)

    # compute the score matrix for the chosen parameters
    for C_power in C_power_range:
        C = 10**C_power

        if gamma_power_range is not None:
            score_C_fixed, loss_C_fixed = [], []

            # compute a row of scores where only gamma can vary, C is fixed
            for gamma_power in gamma_power_range:
                gamma = 10**gamma_power

                loss, score = score_function(C=C, gamma=gamma)

                if loss < best_loss:
                    best_loss, best_score, best_C, best_gamma = loss, score, C, gamma

                # add a value to the row
                score_C_fixed.append(score)
                loss_C_fixed.append(loss)

            # add a computed row to the matrix
            score_matrix.append(score_C_fixed)
            loss_matrix.append(loss_C_fixed)

        else:
            loss, score = score_function(C=C, gamma=None)

            if loss < best_loss:
                best_loss, best_score, best_C = loss, score, C

            score_matrix.append(score)
            loss_matrix.append(loss)

    plt.figure(figure)
    plt.suptitle(str(kernel))
    if gamma_power_range is not None:
        # plot the colormap from the given matrix
        plot_colormap(x_list=C_list, y_list=gamma_list, score_matrix=score_matrix,
                      title='accuracy', subplot=(1, 2, 1),
                      score_min=np.min(score_matrix))
        plot_colormap(x_list=C_list, y_list=gamma_list, score_matrix=loss_matrix,
                      title='loss', subplot=(1, 2, 2),
                      score_max=np.max(loss_matrix), score_min=np.min(loss_matrix))

        print('best result: %.4f for %s, C = %f and gamma = %f, loss = %f'
              %(best_score, str(kernel), best_C, best_gamma, best_loss))

    else:
        # plot the accuracy curve
        plt.subplot(1,2,1)
        plt.plot(C_list, score_matrix, label=str(kernel))
        plt.xscale('log')
        plt.title('accuracy')
        plt.subplot(1,2,2)
        plt.plot(C_list, loss_matrix, label=str(kernel))
        plt.xscale('log')
        plt.title('loss')
        print('best result: %.4f for %s, C = %f, loss = %f'
              % (best_score, str(kernel), best_C, best_loss))

    return best_loss, best_score, best_C, best_gamma



def plot_elbowplot(descriptors, n_cluster_power_range=np.arange(0, 5, 0.5), n_init=5):
    """ plot an elbowplot of the inertia function of the number of cluster
        aims at finding the best number of clusters for the K-means algorithms
        """

    list_inertia = []
    for power in n_cluster_power_range:
        n_clusters = int(10 ** power)

        # creation of the clusters
        KMeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, init_size=5 * n_clusters, n_init=n_init)

        # K-means clustering training
        train_conc = np.concatenate(descriptors)
        KMeans.fit(train_conc)

        list_inertia.append(KMeans.inertia_)

    list_nwords = 10 ** n_cluster_power_range

    # visualisation
    fig = plt.figure()
    plt.plot(list_nwords, list_inertia, 'b-o')
    plt.title('Elbowplot')
    plt.xlabel('n_words')
    plt.ylabel('Inertia')

    return fig
