from sklearn import svm, model_selection, neural_network, neighbors, metrics, cluster, preprocessing
from sklearn.metrics.pairwise import chi2_kernel, sigmoid_kernel, laplacian_kernel
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import random as rd

if __name__ == '__main__':
    from cross_V import descriptors_CV, score_CV, K_fold_cv_split, kmeans_cv

path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"


def display_false(X_image_test, Y_test, Y_test_pred):
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    plt.figure()
    c = 0
    n=0
    while c<24:
        n+=1
        if Y_test_pred[n] != Y_test[n]:
            c+=1
            print(c)
            plt.subplot(4, 6, c, xticks=[], yticks=[])
            plt.imshow(X_image_test[n], cmap='gray')
            plt.text(0.1,0.1,str(classes[Y_test_pred[n]-1])+' / '+str(classes[Y_test[n]-1]),fontsize=6,bbox=dict(facecolor='red', alpha=1))
    plt.show()



def display_(X_image_test, Y_test, Y_test_pred):
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    plt.figure()
    c=0
    while c<40:
        c += 1
        n=rd.randint(0,500)
        plt.subplot(5, 8, c, xticks=[], yticks=[])
        plt.imshow(X_image_test[n], cmap='gray')
        if Y_test_pred[n] != Y_test[n]:
            plt.text(0.1,0.1,str(classes[Y_test_pred[n]-1]),fontsize=6,bbox=dict(facecolor='red', alpha=1))
        else:
            plt.text(0.1,0.1,str(classes[Y_test_pred[n]-1]),fontsize=6,bbox=dict(facecolor='white', alpha=1))
    plt.show()

def descriptors(path, test_proportion=0.1, nb_class=6, lib="SURF"):
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

    descriptors_train = []
    descriptors_test = []
    X_image_test = []

    Y_train = []
    Y_test = []

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
            img_gray = np.mean(img, -1)

            # on transforme l'image en array dtype='uint8'
            img_uint8 = np.uint8(img_gray)

            # les keypoints (coordonnées des patches) et descripteurs
            kp, des = extrac.detectAndCompute(img_uint8, None)

            # on cree les descripteurs d'apprentissage et de test dans les proportions choisies
            if i < nb_img * (1-test_proportion):
                # creation des listes qui construiront la base d'apprentissage
                descriptors_train.append(des)
                Y_train.append(classe)

            else:
                # creation des listes qui contruiront la base de test
                descriptors_test.append(des)
                Y_test.append(classe)
                X_image_test.append(img)

    print("--- Done")
    return descriptors_train, descriptors_test, Y_train, Y_test, X_image_test


# descriptors_train, descriptors_test, Y_train, Y_test = descriptors(path)
#
# flat_descriptors = np.concatenate(descriptors_train)
#
# KMeans = cluster.MiniBatchKMeans(n_clusters=500, init_size=3*500)
# KMeans.fit(flat_descriptors)


def train_generation(descriptors_train, KMeans):
    """generation de la base d'apprentissage
    INPUT : list of list of descriptors
    OUTPUT : X vector to use for the training (normalised and clustered)
    """
    #print("--- creation de la base d'entrainement :")

    n_words = KMeans.n_clusters
    X_train =[]
    start=0
    # chaque image est associee à un codeword (liste des fréquences d'apparition des mots visuels)
    for num_img, descriptors_img in enumerate(descriptors_train, 0):
        # on initialise l'histogramme a 0
        words_histogram = np.zeros(n_words)
        nb_des = len(descriptors_img)

        # on ajoute ensuite les mots correspondants aux descripteurs
        associated_word = \
            KMeans.labels_[start:start + nb_des]

        # KMeans.predict(descriptors_img)
        # # on indente l'indice de debut de la liste de descripteurs pour passer au mot suivant
        start += nb_des
        for word in associated_word:
            words_histogram[word] += 1/nb_des # pour normaliser la somme a 1

        X_train.append(words_histogram)
    return X_train


# X_train = train_generation(descriptors_train, KMeans)


def test_generation(descriptors_test, KMeans):
    """generation de la base de test
    INPUT : list of list of descriptors
    OUTPUT : X vector to use for the test(normalised and clustered)
    """
    #print("--- creation de la base de test:")

    n_words = KMeans.n_clusters
    X_test =[]
    # chaque image est associee à un codeword (liste des fréquences d'apparition des mots visuels)
    for num_img, descriptors_img in enumerate(descriptors_test, 0):
        # on initialise l'histogramme a 0
        words_histogram = np.zeros(n_words)
        nb_des = len(descriptors_img)

        # on ajoute ensuite les mots correspondants aux descripteurs
        associated_word = KMeans.predict(descriptors_img)
        for word in associated_word:
            words_histogram[word] += 1/nb_des # pour normaliser la somme a 1

        X_test.append(words_histogram)

    return X_test


# X_test = test_generation(descriptors_test, KMeans)


def learn_SVM(X_train, X_test, Y_train, Y_test, X_image_test, kernel='linear', C=1, gamma=None,  print_result=True, display=False):
    """entraine un modele SVM a partir des donnes
    INPUT : listes train/test
    OUTPUT : score
    """

    # entrainement du modele SVM
    if gamma is not None:
        SVM_model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    else:
        SVM_model = svm.SVC(kernel=kernel, C=C)
    SVM_model.fit(X_train, Y_train)
    score = SVM_model.score(X_test, Y_test)

    if print_result:
        if gamma is None:
            print("score SVM %s %.3f ,  C = %.4f" % (kernel, score, C))
        else:
            print("score SVM %s %.3f ,  C = %.4f, gamma = %.5f" % (kernel, score, C, gamma))
        Y_test_pred = SVM_model.predict(X_test)
        conf_matrix = metrics.confusion_matrix(Y_test, Y_test_pred)
        print(conf_matrix)

        if display:
            display_(X_image_test,Y_test,Y_test_pred)

    else:
        conf_matrix=None

    return score, conf_matrix


def score_colormap_C_gamma(path, n_words=100, test_proportion=0.1):
    """try to find the best parameters gamma and C the same time
    I use here a color map representing the score"""
    print("### score_colormap")

    # creation des descripteurs
    des_train, des_test, Y_train, Y_test, X_image_test = descriptors(path, test_proportion, lib="SURF")

    # clustering par unsupervised-learning, creation du vocabulaire
    KMeans = cluster.MiniBatchKMeans(n_clusters=n_words, init_size=3*n_words, n_init=5)  # utiliser MiniBatchKmeans plutôt que KMeans
    train_conc = np.concatenate(des_train)
    KMeans.fit(train_conc)

    # creation des jeux de test et d'entrainement
    X_train = train_generation(des_train, KMeans)
    X_test = test_generation(des_test, KMeans)

    score_linear = []
    score_rbf = []
    score_chi2 = []
    score_lap = []
    score_max = 0
    score_min = 1

    list_C = range(-4, 6)
    list_gamma = range(-5, 5)
    for p in list_C:
        C = 10 ** p

        S_lin = []
        S_rbf = []
        S_chi = []
        S_lap = []

        for q in list_gamma:
            gamma = 10 ** q

            S_lin.append(learn_SVM(X_train, X_test, Y_train, Y_test, X_image_test, kernel='linear', C=C))
            S_rbf.append(learn_SVM(X_train, X_test, Y_train, Y_test, X_image_test, kernel='rbf', gamma=gamma, C=C))
            S_chi.append(learn_SVM(X_train, X_test, Y_train, Y_test, X_image_test, kernel=chi2_kernel, C=C, gamma=gamma,
                                   print_result=True, display=False))
            S_lap.append(learn_SVM(X_train, X_test, Y_train, Y_test, X_image_test, kernel=laplacian_kernel, C=C, gamma=gamma,
                                   print_result=True, display=False))

            for list_score in [S_chi,S_lin, S_rbf, S_lap]:
                if max(list_score) > score_max:
                    score_max = max(list_score)
                if min(list_score) < score_min:
                    score_min = min(list_score)


        score_linear.append(S_lin)
        score_rbf.append(S_rbf)
        score_chi2.append(S_chi)
        score_lap.append(S_lap)

    plt.subplot(2,2,1)
    plt.pcolor(list_C, list_gamma, np.transpose(score_linear), cmap='RdBu_r', norm=colors.LogNorm()
               , vmax=score_max, vmin=score_min)
    plt.title("linear")
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.pcolor(list_C, list_gamma, np.transpose(score_rbf), cmap='RdBu_r', norm=colors.LogNorm(),
               vmax=score_max, vmin=score_min)
    plt.title("rbf")
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.pcolor(list_C, list_gamma, np.transpose(score_chi2), cmap='RdBu_r', norm=colors.LogNorm(),
               vmax=score_max, vmin=score_min)
    plt.title("chi2")
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.pcolor(list_C, list_gamma, np.transpose(score_lap), cmap='RdBu_r', norm=colors.LogNorm(),
               vmax=score_max, vmin=score_min)
    plt.title("laplacian")
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def score_colormap_C_nwords(path, nb_fold=5):
    """try to find the best parameters n_words and C the same time
    I use here a color map representing the score"""
    print("### score_colormap")

    # creation des descripteurs pour de la CV
    list_descriptors_cv, list_Y_cv, list_img = descriptors_CV(path, nb_class=6)
    #des_train, des_test, Y_train, Y_test, X_image_test = descriptors(path, test_proportion, nb_class=6, lib="SURF")

    score_chi2 = []
    variance_chi2 = []

    list_C = np.linspace(0, 4, 17)
    list_nwords = np.linspace(500, 3000, 11)

    database_descriptors = (list_descriptors_cv, list_Y_cv)
    list_train_sets, list_test_sets = K_fold_cv_split(database_descriptors, nb_fold=nb_fold)

    for n in list_nwords:
        n_words = int(n)
        print("__________________________ n_words =", n_words)

        list_kmeans = kmeans_cv(nb_fold, list_train_sets, n_words)

        # # clustering par unsupervised-learning, creation du vocabulaire
        # KMeans = cluster.MiniBatchKMeans(n_clusters=n_words, init_size=3 * n_words,
        #                                  n_init=5)  # utiliser MiniBatchKmeans plutôt que KMeans
        # train_conc = np.concatenate(des_train)
        # KMeans.fit(train_conc)
        #
        # # creation des jeux de test et d'entrainement
        # X_train = train_generation(des_train, KMeans)
        # X_test = test_generation(des_test, KMeans)

        S_chi = []
        V_chi = []

        for p in list_C:
            C = 10 ** p
            print("C = %.2f" %C)

            S,V = score_CV(list_kmeans, list_train_sets, list_test_sets, nb_fold=5, C=C)
            # S = learn_SVM(X_train, X_test, Y_train, Y_test, kernel=chi2_kernel, C=C, print_result=True)
            S_chi.append(S)
            V_chi.append(V)
            # gamma is chosen automatically

        score_chi2.append(S_chi)
        variance_chi2.append(V_chi)

    # list_C = 10 ** np.array(list_C)

    std_chi2 = np.sqrt(variance_chi2)

    fig = plt.figure(1)

    ax1 = fig.add_subplot(121)
    plt.pcolormesh(list_C, list_nwords, score_chi2, cmap='RdBu_r', norm=colors.LogNorm(), shading='gouraud')
    ax1.title.set_text("chi2 score")
    plt.xlabel("log10(C)")
    plt.ylabel("log10(n_words)")
    plt.colorbar(pad=0.01)

    ax2 = fig.add_subplot(122)
    plt.pcolormesh(list_C, list_nwords, std_chi2, cmap='RdBu_r', norm=colors.LogNorm(), shading='gouraud')
    ax2.title.set_text("chi2 std")
    plt.xlabel("log10(C)")
    plt.ylabel("log10(n_words)")
    plt.colorbar(pad=0.01)

    fig2 = plt.figure(2)

    ax3 = fig2.add_subplot(121)
    plt.pcolor(list_C, list_nwords, score_chi2, cmap='RdBu_r', norm=colors.LogNorm())
    ax3.title.set_text("chi2 score")
    plt.xlabel("log10(C)")
    plt.ylabel("log10(n_words)")
    plt.colorbar()

    ax4 = fig2.add_subplot(122)
    plt.pcolor(list_C, list_nwords, std_chi2, cmap='RdBu_r', norm=colors.LogNorm())
    ax4.title.set_text("chi2 std")
    plt.xlabel("log10(C)")
    plt.ylabel("log10(n_words)")
    plt.colorbar()

    plt.show()


def SVM_model_single(path, n_words=1000, C=1, gamma=None, kernel='linear',n_class=6, n_fold=5, init_cluster=100):
    # creation des descripteurs
    des_train, des_test, Y_train, Y_test, X_image_test = descriptors(path, 1/n_fold, lib="SURF", nb_class=n_class)

    # clustering par unsupervised-learning, creation du vocabulaire
    KMeans = cluster.MiniBatchKMeans(n_clusters=n_words, init_size=3 * n_words,
                                     n_init=init_cluster)  # utiliser MiniBatchKmeans plutôt que KMeans
    train_conc = np.concatenate(des_train)
    KMeans.fit(train_conc)

    # creation des jeux de test et d'entrainement
    X_train = train_generation(des_train, KMeans)
    X_test = test_generation(des_test, KMeans)

    result = learn_SVM(X_train, X_test, Y_train, Y_test, X_image_test,
                      kernel=kernel, gamma=gamma, C=C, display=True)

    print('score : ', result[0])

    print('confusion matrix : \n', result[1])

    return result


# TODO: optional arguments
def SVM_model_CV(path, n_words=1000, C=1, gamma=None, kernel='linear',n_class=6, n_fold=5, init_cluster=100):

    # creation des descripteurs pour de la CV
    list_descriptors_cv, list_Y_cv, list_img = descriptors_CV(path, nb_class=n_class)

    database_descriptors = (list_descriptors_cv, list_Y_cv)
    list_train_sets, list_test_sets = K_fold_cv_split(database_descriptors, nb_fold=n_fold)

    list_kmeans = kmeans_cv(n_fold, list_train_sets, n_words, n_init=init_cluster)

    score, variance, TCM = score_CV(list_kmeans, list_train_sets, list_test_sets,
                                    nb_fold=n_fold, C=C, print_result=True, display=False)

    print(score, variance)
    print(TCM)

    #os.system('pause')
    normed_TCM = TCM/np.sum(TCM, axis=1)
    class_names = os.listdir(path)
    print(class_names)

    plt.figure(0)
    plt.title("SVM total confusion matrix")
    plot_conf_matrix = sns.heatmap(normed_TCM, annot=TCM, fmt="d", cmap='Blues',
                                   xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Class')
    plt.show()


if __name__ == '__main__':
    # score_colormap_C_nwords(path)
    # score_colormap_C_gamma(path, n_words=1000, test_proportion=0.2)
    # SVM_model_CV(path, n_words=1000, C=4, gamma=None, n_fold=2, n_class=2, init_cluster=1)
    SVM_model_single(path, n_words=1000, C=4, gamma=None, kernel=chi2_kernel, n_class=6, n_fold=5, init_cluster=100)
