from sklearn import svm
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision

import cv2

#image = io.imread(PATH_TO_THE_IMAGE)
#img = transform.resize(image, (desired_h, desired_w))

path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"

directory = os.listdir(path)[4]
file = os.listdir(path + directory)[8]

# ouverture de l'image et passage en niveau de gris
img_color = plt.imread(path + directory + '/' + file)
img_small = transform.resize(img_color, (224, 224))
print("shape : ", img_color.shape)
print("shape small: ", img_small.shape)

img_red = img_color[:,:,0]
img_green = img_color[:,:,1]
img_blue = img_color[:,:,2]

img_gray = np.mean(img_color, -1)

plt.figure(1)
plt.imshow(img_small)
plt.figure(2)
plt.imshow(img_color)
plt.show()