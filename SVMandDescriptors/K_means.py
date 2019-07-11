from sklearn import cluster
import numpy as np


def kmeans_clustering(descriptor_train, n_clusters=1000):
    """ Unsupervised clustering of the descriptors """

    KMeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, init_size=3*n_clusters)  # use MiniBatchKmeans preferably to KMeans
    train_conc = np.concatenate(descriptor_train)
    KMeans.fit(train_conc)

    return KMeans


def train_generation(descriptors_train_shape, n_words, KMeans):
    """ Training base generation from the set of descriptors that describes the images """

    X_train = []
    current_start_idx = 0
    # each images is describe as a list of frequencies associated to the categories of descriptors
    for i, length_image_i in enumerate(descriptors_train_shape):
        # initialisation of the histogram
        words_histogram = np.zeros(n_words)

        # adding the descriptors one by one to the histogram
        current_length = length_image_i
        associated_words = KMeans.labels_[current_start_idx: current_start_idx + current_length]
        for j, word in enumerate(associated_words):
            # divide by the length to normalize the sum to 1
            words_histogram[word] += 1/current_length

        # all the images are now a set of frequencies, all the same length
        X_train.append(words_histogram)

        # update the start index of the list
        current_start_idx += current_length

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
        associated_words = KMeans.predict(descriptors_image)
        for j, word in enumerate(associated_words):
            # divide by the length to normalize the sum to 1
            words_histogram[word] += 1 / length
            # all the images are now a set of frequencies, all the same length

        X_val.append(words_histogram)

    return X_val
