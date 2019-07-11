import matplotlib.pyplot as plt
import numpy as np
from SVMandDescriptors.SVM_models import learn_SVM
from SVMandDescriptors.Descriptors import descriptors_all_base
from SVMandDescriptors.K_means import kmeans_clustering, train_generation, val_generation
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import chi2_kernel
from SVMandDescriptors.Results_visualization import display_test_images


# path to the dataset
path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"

# model parameters
TEST_SIZE = 0.3
N_CLUSTERS = 1000
KERNEL = chi2_kernel
C = 4
GAMMA = 1


# ---------- description of images with descriptors ---------- #
print('-'*10 + ' descriptors extraction ' + '-'*10)

# descriptors extraction
list_descriptors, list_y, list_img = descriptors_all_base(path, nb_class=6, method="SURF")

# split database into train and validation
indices = range(len(list_y))
descriptors_train, descriptors_val,\
y_train, y_val, idx_train, idx_val = train_test_split(list_descriptors, list_y, indices,
                                                      test_size=TEST_SIZE, shuffle=True)

# a list containing the numer of descriptors for each image
# this is to read directly the clustered points without having to predict on training points
descriptors_train_shape = [len(descriptors_train[img_k]) for img_k, des_img_k in enumerate(descriptors_train)]

# ---------- creation of the clusters and unsupervised training ---------- #
print('-'*10 + ' K-means clustering ' + '-'*10)

# K-means clustering
K_means = kmeans_clustering(descriptors_train, n_clusters=N_CLUSTERS)

# features generation
X_train = train_generation(descriptors_train_shape, n_words=N_CLUSTERS, KMeans=K_means)
X_val = val_generation(descriptors_val, n_words=N_CLUSTERS, KMeans=K_means)


# ---------- training the SVM model ---------- #
print('-'*10 + ' SVM optimization ' + '-'*10)

# optimization on the training set
SVM_model, loss, score, confusion_matrix = learn_SVM(X_train, y_train, X_val, y_val,
                                                     kernel=KERNEL, C=C, gamma=GAMMA,
                                                     print_result=True, print_all=True)


# ---------- visualization of a sample---------- #

# selection of the tested images
X_image_test = np.array(list_img)[idx_val]

# prediction on the validation set
y_predict = SVM_model.predict(X_val)

# display
display_test_images(X_image_test, y_val, y_predict)
plt.show()
