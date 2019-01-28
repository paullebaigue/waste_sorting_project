# from sklearn import svm
# from skimage import io, transform
import numpy as np
import time
import random as rd
# import matplotlib.pyplot as plt
# import os
# import torch
# import torchvision
#
# import cv2
#
# #image = io.imread(PATH_TO_THE_IMAGE)
# #img = transform.resize(image, (desired_h, desired_w))
#
# path = "C:/Users/OrdiPaul/Documents/Mines Nancy/Projet/Projet3A_wastesorting/dataset/"
#
# directory = os.listdir(path)[4]
# file = os.listdir(path + directory)[8]
#
# # ouverture de l'image et passage en niveau de gris
# img_color = plt.imread(path + directory + '/' + file)
# img_small = transform.resize(img_color, (224, 224))
# print("shape : ", img_color.shape)
# print("shape small: ", img_small.shape)
#
# img_red = img_color[:,:,0]
# img_green = img_color[:,:,1]
# img_blue = img_color[:,:,2]
#
# img_gray = np.mean(img_color, -1)
#
# plt.figure(1)
# plt.imshow(img_small)
# plt.figure(2)
# plt.imshow(img_color)
# plt.show()


# taille_pop = 1000
# best = round(taille_pop*1/6)
# random = round(taille_pop*1/12)
# parents = best + random
# enfant = parents * 3
# new_pop = parents + enfant
# print(best,
#       random,
#       parents,
#       enfant,
#       new_pop)




def f(n):
    print(n)
    return n

def wrap_func(L):
    def func(n):
        time.sleep(1/(n+1))
        print(n)
        return n,n**2
    return func


# if __name__ == '__main__':
#     pool = multiprocessing.Pool(processes=4)
#     pool.map(wf, range(10))
#     pool.close()
#     pool.join()
#     print('done')

import multiprocessing
from joblib import Parallel, delayed

inputs = range(10)
L=[]
num_cores = multiprocessing.cpu_count()
print("num_cores : ", num_cores)
list_result = Parallel(n_jobs=num_cores)(delayed(wrap_func(L))(i) for i in inputs)
N1,N2 = np.transpose(list_result)
print(N1,N2)