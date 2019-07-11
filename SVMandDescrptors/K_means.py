from sklearn import cluster
import numpy as np


def kmeans_clustering(descriptor_train, n_clusters=1000):
    """ Unsupervised clustering of the descriptors """

    KMeans = cluster.MiniBatchKMeans(n_clusters=n_clusters)  # use MiniBatchKmeans preferably to KMeans
    train_conc = np.concatenate(descriptor_train)
    KMeans.fit(train_conc)

    return KMeans


def train_generation(idx_descriptors_train, n_words, KMeans):
    """ Training base generation from the set of descriptors that describes the images """

    X_train =[]
    # each images is describe as a list of frequencies associated to the categories of descriptors
    for i, indexes_image in enumerate(idx_descriptors_train):
        # initialisation of the histogram
        words_histogram = np.zeros(n_words)

        # adding the descriptors one by one to the histogram
        length = len(indexes_image)
        for j, idx_single_descriptor in enumerate(indexes_image):
            associated_word = KMeans.labels_[idx_single_descriptor]
            # divide by the length to normalize the sum to 1
            words_histogram[associated_word] += 1/length

        # all the images are now a set of frequencies, all the same length
        X_train.append(words_histogram)

    return X_train


def val_generation(descriptors_test, n_words, KMeans):
    """ Validation base generation from the set of descriptors that describes the images """

    X_val = []
    # each images is describe as a list of frequencies associated to the categories of descriptors
    for i, descriptors_image in enumerate(descriptors_test):
        # initialisation of the histogram
        words_histogram = np.zeros(n_words)

        # adding the descriptors one by one to the histogram
        length = len(descriptors_image)
        for j, single_descriptor in enumerate(descriptors_image):
            associated_word = KMeans.predict(single_descriptor)
            # divide by the length to normalize the sum to 1
            words_histogram[associated_word] += 1 / length

            # all the images are now a set of frequencies, all the same length
        X_val.append(words_histogram)

    return X_val
