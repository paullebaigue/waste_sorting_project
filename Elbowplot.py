import numpy as np
from image_SVM_classifier import descriptors
from sklearn import cluster
import matplotlib.pyplot as plt

path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"

# creation des descripteurs
des_train, des_test, Y_train, Y_test, X_image_test = descriptors(path, lib="SURF")

step = 0.1
list_nwords = np.arange(0.6, 4.3, step)
list_inertia = []
for p in list_nwords:
    n_words = int(10**p)
    print(n_words)
    # creation des cluster de bags of words
    KMeans = cluster.MiniBatchKMeans(n_clusters=n_words, init_size=3*n_words, n_init=50)

    # entrainement du cluster
    train_conc = np.concatenate(des_train)
    KMeans.fit(train_conc)

    list_inertia.append(KMeans.inertia_)

plt.figure()
plt.plot(list_nwords, list_inertia, 'b-o')
plt.title('Elbowplot')
plt.xlabel('log10(n_words)')
plt.ylabel('Inertia')
plt.show()


