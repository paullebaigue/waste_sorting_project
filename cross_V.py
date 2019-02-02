import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import svm, model_selection, neural_network, neighbors, metrics, cluster, preprocessing
from sklearn.metrics.pairwise import chi2_kernel
from image_SVM_classifier import train_generation, test_generation, learn_SVM
import multiprocessing
from joblib import Parallel, delayed


path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"


def K_fold_cv_split(data_base, nb_fold=10):
    """cree des base d'entrainement et de test pour faire une cross validation
    INPUT : base de donnees (X,Y)
    OUTPUT : listes des bases train/test
    """

    # on met au bon format pour rendre le code robust
    if len(data_base) == 2:
        data_base = np.array((list(data_base[0]), data_base[1]))
        data_base = np.transpose(np.array(data_base))
    data_shuffle = list(data_base)
    # on mélange la liste pour garantir la repartition aleatoire
    random.shuffle(data_shuffle)
    list_train_sets = []
    list_test_sets = []
    L = len(data_shuffle)

    for n in range(nb_fold):
        # on cree les K_folds
        train = data_shuffle[: int(n/nb_fold*L)] + data_shuffle[int((n+1)/nb_fold*L) :]
        test = data_shuffle[int(n/nb_fold*L) : int((n+1)/nb_fold*L)]

        list_train_sets.append(train)
        list_test_sets.append(test)

    return list_train_sets, list_test_sets

# Y = list(np.zeros(4)) + list(np.ones(4))
# X = np.random.rand(8, 2)
# database = (X,Y)
# # # database = [[[1,2], 0], [[1,2], 0], [[1,2], 0], [[1,2], 0], [[1,2], 0], [[1,2], 0], [[1,2], 0], [[1,2], 0]]
# list_train_sets, list_test_sets = K_fold_cv_split(database, nb_fold=3)
# # print(np.transpose(list_train_sets))
# # print(np.transpose(list_test_sets))
#
# [X_train, Y_train] = np.transpose(list_train_sets[0])
#
# print(len(list_train_sets))
# print(X_train)
# print(Y_train)


# TODO: parloop
def descriptors_CV(path, nb_class=6, lib="SURF"):
    """creation des descripteurs a partir des images
    INPUT : chemin d'acces
    OUTPUT : listes des descripteurs (X), listes des Y
    """
    print("--- extraction des descripteurs :")

    # creation d'un createur de descripteur 2D
    if lib == "SURF":
        extrac = cv2.xfeatures2d.SURF_create()
    elif lib == "SIFT":
        extrac = cv2.xfeatures2d.SIFT_create()
    else:
        raise ValueError('descriptor extractor not supported, try "SIFT" or "SURF" ')

    list_descriptors_cv = []
    list_Y_cv = []

    classe = 0
    # on parcourt toutes les images du fichier
    for directory in os.listdir(path)[6-nb_class:]:
        classe += 1
        print("classe %d : %s" % (classe, directory))

        nb_img = len(os.listdir(path + directory))
        i = 0
        for file in os.listdir(path + directory):
            i+=1

            # ouverture de l'image et passage en niveau de gris
            img = plt.imread(path + directory + '/' + file)
            img = np.mean(img, -1)

            # on transforme l'image en array dtype='uint8'
            img_uint8 = np.uint8(img)

            # les keypoints (coordonnées des patches) et descripteurs
            kp, des = extrac.detectAndCompute(img_uint8, None)

            # on cree les descripteurs
            # creation des listes qui construiront la base d'apprentissage
            list_descriptors_cv.append(des)
            list_Y_cv.append(classe)

    print("--- Done")
    return list_descriptors_cv, list_Y_cv


def score_CV(list_kmeans, list_train_sets, list_test_sets, nb_fold=5, C=1, print_result=False):


    assert nb_fold == len(list_kmeans)

    list_score = []
    list_conf_matrix = []
    for k in range(nb_fold):

        [descriptor_train, Y_train] = np.transpose(list_train_sets[k])
        [descriptor_test, Y_test] = np.transpose(list_test_sets[k])
        KMeans = list_kmeans[k]

        # # clustering par unsupervised-learning, creation du vocabulaire
        # KMeans = cluster.MiniBatchKMeans(n_clusters=n_words, init_size=3 * n_words,
        #                                  n_init=5)  # utiliser MiniBatchKmeans plutôt que KMeans
        # train_conc = np.concatenate(descriptor_train)
        # KMeans.fit(train_conc)

        X_train = train_generation(descriptor_train, KMeans)
        Y_train = list(Y_train)
        Y_test = list(Y_test)
        X_test = test_generation(descriptor_test, KMeans)
        score_k, conf_matrix_k = learn_SVM(X_train, X_test, Y_train, Y_test, "no_X_img", kernel=chi2_kernel, C=C,
                            print_result=print_result)
        list_score.append(score_k)
        list_conf_matrix.append(conf_matrix_k)
        # if print_result:
        print("score-%d = %.4f " %(k+1, score_k))

    score = np.mean(list_score)
    var = np.var(list_score)

    #print(list_conf_matrix)
    TCM = np.sum(list_conf_matrix, axis=0)

    return score, var, TCM


def wrap_kmeans(list_train_sets, n_words, n_init=10):
    def kmeans_k(k):
        [descriptor_train, Y_train] = np.transpose(list_train_sets[k])

        kmeans = cluster.MiniBatchKMeans(n_clusters=n_words, init_size=5*n_words, n_init=n_init)
        train_conc = np.concatenate(descriptor_train)
        kmeans.fit(train_conc)
        return kmeans
    return kmeans_k


def kmeans_cv(nb_fold, list_train_sets, n_words, n_init=10):
        num_cores = 2
        kmeans_k = wrap_kmeans(list_train_sets, n_words, n_init=n_init)
        list_kmeans = Parallel(n_jobs=num_cores)(delayed(kmeans_k)(k) for k in range(nb_fold))
        return list_kmeans

if __name__ == '__main__':
    list_descriptors_cv, list_Y_cv = descriptors_CV(path, nb_class=2)
    print("score CV : ", np.around(score_CV(list_kmeans, nb_fold=5, C=10, print_result=True),4))

