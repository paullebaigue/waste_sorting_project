import matplotlib.pyplot as plt
import numpy.random as rd
import numpy as np
from sklearn import cluster


def display_test_images(X_image_test, Y_test, Y_test_pred):
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    plt.figure()
    c=0
    while c<40:
        c += 1
        n=rd.randint(0,500)
        plt.subplot(5, 8, c, xticks=[], yticks=[])
        plt.imshow(X_image_test[n], cmap='gray')
        if Y_test_pred[n] != Y_test[n]:
            plt.text(0.1, 0.1, str(classes[Y_test_pred[n]-1]),
                    fontsize=6, bbox=dict(facecolor='red', alpha=1))
        else:
            plt.text(0.1, 0.1, str(classes[Y_test_pred[n]-1]),
                     fontsize=6, bbox=dict(facecolor='white', alpha=1))
    return


def colormap(kernel='linear', score_function='function'):
    return


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
    plt.figure()
    plt.plot(list_nwords, list_inertia, 'b-o')
    plt.title('Elbowplot')
    plt.xlabel('n_words')
    plt.ylabel('Inertia')

    return
