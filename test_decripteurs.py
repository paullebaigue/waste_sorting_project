import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

path="C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"


"""Affichage d'une image et l'histogramme de ses niveaux de gris"""
def hist_img_gray_example(plot_img=False, print_size=False, plot_hist=False, number=0):
    # ouverture d'une image
    file_1= os.listdir(path)[4]
    img_1 = os.listdir(path + file_1)[number]
    img_1 = Image.open(path + file_1 + '/' + img_1)
    # conversion en niveaux de gris
    img_1_gray = img_1.convert('L')

    # transformation en np array
    img_1 = np.asarray(img_1)
    img_1_gray = np.asarray(img_1_gray)

    # affichage de l'image
    if plot_img:
        fig = plt.figure(221)
        ax1 = plt.subplot(2,2,1)
        plt.imshow(img_1)
        ax2 = plt.subplot(2,2,2)
        plt.imshow(img_1_gray, cmap='gray')
        ax3 = plt.subplot(2,2,3)
        plt.imshow(img_1_gray)
        ax4 = plt.subplot(2,2,4)
        plt.imshow(img_1_gray, cmap='coolwarm')
        plt.show()

    # mise en ligne de l'array
    array_gray = img_1_gray.reshape(-1)
    if print_size:
        print(len(img_1_gray), len(img_1_gray[0]))
        print(array_gray, len(array_gray))

    # affichage de l'histogramme
    if plot_hist:
        plt.figure(211)
        plt.subplot(211)
        plt.hist(array_gray, bins=40, density=True, range=(0,255))
        plt.subplot(212)
        plt.hist(array_gray, bins=40, density=True, range=(5,250))
        plt.show()
        hist = np.histogram(array_gray, density=True, bins=5)
        print(hist)

    return

number=20
#hist_img_gray_example(plot_img=True, print_size=False, plot_hist=True, number=number)

"""creation de descripteurs"""

# ouverture d'une image
file_1 = os.listdir(path)[0]
img_1 = os.listdir(path + file_1)[number]
img_1 = Image.open(path + file_1 + '/' + img_1)


surf = cv2.xfeatures2d.SURF_create(3000)
# descriptors = np.empty([0,64],dtype='uint8')   # bag of visual words
# idx_descriptors = []   # indice des descripteurs stockés dans descriptors

# on transforme l'image en array dtype='uint8'
print(img_1.size, img_1.size[0]*img_1.size[1])
img_uint8 = np.uint8(img_1)

kp, des = surf.detectAndCompute(img_uint8, None)  # les keypoints (coordonnées des patches) et descripteurs

img2 = cv2.drawKeypoints(img_uint8,kp,None,flags=4)
plt.imshow(img2),plt.show()
print(len(kp), len(des), img_1.size)