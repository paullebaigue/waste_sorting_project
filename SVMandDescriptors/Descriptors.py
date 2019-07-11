import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def descriptors_single_img(path, method="SURF", *args):
    """extract all the descriptors of a single image"""

    # choice of the 2D descriptors extractor
    if method == "SURF":
        extractor = cv2.xfeatures2d.SURF_create()
    elif method == "SIFT":
        extractor = cv2.xfeatures2d.SIFT_create()
    else:
        raise ValueError('descriptor extractor not supported, try either "SIFT" or "SURF" ')

    # open image in gray scale
    img = plt.imread(path)
    img_gray = np.mean(img, -1)

    # transform image into array dtype='uint8'
    img_uint8 = np.uint8(img_gray)

    # key-points (patches coordinates) and descriptors
    key_points, descriptors = extractor.detectAndCompute(img_uint8, None)

    return descriptors, img


# TODO: parloop
def descriptors_all_base(path, nb_class=6, method="SURF", *args):
    """creation of descriptors from images of the base
    create a list of descriptors and targets, describing all the images of the base
    OUTPUT : list of lists of descriptors (X), list of lists of targeted classes (Y)
    """

    list_descriptors = []
    list_y = []
    list_img = []

    class_img = 0
    # iterating over the categories of images of the database
    for directory in os.listdir(path)[6-nb_class:]:
        class_img += 1
        print("class %d : %s" % (class_img, directory))

        # iterating over the images of a category
        for file in os.listdir(path + directory):

            # extraction of the list of descriptors of a single image
            descriptors, img = descriptors_single_img(path + directory + '/' + file, method=method)

            # lists of descriptors which will be the training sets
            list_descriptors.append(descriptors)
            list_y.append(class_img)
            list_img.append(img)

    return list_descriptors, list_y, list_img


def descriptors_per_image_indexes(descriptor_train):
    """ return the lists of the descriptor indexes associated to every single image"""

    # initialize the right amount of images
    idx_descriptors_train = np.empty(len(descriptor_train))

    start = 0
    # iteration over all the images
    for i, image_described in descriptor_train:

        # add the indexes to the list
        nb_descriptors = len(image_described)
        idx_descriptors_train[i] = range(start, nb_descriptors)

        # update the start index
        start += nb_descriptors

    return idx_descriptors_train


