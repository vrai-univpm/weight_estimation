#%%
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import tensorflow as tf
import keras
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPool2D
from keras.optimizers import Adam, Adamax, SGD, RMSprop,Nadam
from keras.models import Model, Sequential
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.preprocessing.image import load_img,img_to_array
import pandas as p
import numpy as np
import os
from keras import metrics
import subprocess
import re
from keras.models import load_model
from scipy import ndimage, misc
from sklearn.model_selection import train_test_split
from keras import regularizers
import skimage
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib
from skimage.transform import rescale,resize,downscale_local_mean
from PIL import Image
import csv
from glob import glob
from keras.models import load_model
#%%
def create_inceptionV3():
    print('Creata')
    return None