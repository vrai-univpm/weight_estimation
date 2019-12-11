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
from datetime import datetime

# Taking best model results:
n_fold = 5

content = './'

savingPath = content + 'WeightCrossValidation/02102019_04_58_17/'

savingWeight = savingPath + 'weights/'

#bestModelFolder = savingPath + 'CrossValidation/'
# model loading
model = load_model(savingPath + 'model200Epoch02102019_04_58_17.h5')
# test array loading
img_np_array_test = np.load(savingPath + 'img_test_set.npy')
lab_np_array_test = np.load(savingPath + 'lab_test_set.npy')
# Selecting the minimum mae from saved weights:
# Initialize a minimum reference
minimum_value = {}
minimum_fileName = {}
# init of dictionary of the minimum
for key in range(0,n_fold):
  minimum_value[key] = 100.000
for file in os.listdir(savingWeight):
    if file.endswith(".h5"):
      name = file.replace('.h5','')
      value = float(name.split('-')[3])
      key = int(name.split('-')[4])
      if value < minimum_value[key]:
        minimum_value[key] = value
        minimum_fileName[key] = file
print(minimum_fileName)
print(minimum_value)
test_mae = {}
statSave = open(savingPath + 'StatisticalAnalysis.txt', 'a+')
statSave.write('\nResults statistical analysis:\n')
statSave.close()
for key in minimum_fileName.keys():
  statSave = open(savingPath + 'StatisticalAnalysis.txt', 'a+')
  statSave.write('\n')
  model.load_weights(savingWeight + minimum_fileName[key])
  #print(model.metrics_names)
  loss, mae, mse = model.evaluate(img_np_array_test, lab_np_array_test, verbose=0)
  test_mae[key] = mae
  print('In fold' + str(key) + '\nTesting set Mean Abs Error: {:5.2f} Kg'.format(mae),'\t Testing Mean Square Error: {:5.2f} kg'.format(mse))
  statSave.write('In fold' + str(key) + '\n\tTesting set Mean Abs Error: {:5.2f} Kg'.format(mae) + '\n\t Testing Mean Square Error: {:5.2f} kg'.format(mse))
  
  pred = model.predict(img_np_array_test)
  
  predictions = model.predict(img_np_array_test).flatten()
  plt.figure(str(key))
  plt.scatter(lab_np_array_test, predictions)
  plt.xlabel('True Values [Kg]')
  plt.ylabel('Predictions [Kg]')
  plt.axis('equal')
  plt.axis('square')
  plt.xlim([0,plt.xlim()[1]])
  plt.ylim([0,plt.ylim()[1]])
  _ = plt.plot([-100, 150], [-100, 150])
  plt.savefig(savingPath + 'pred_inc-' + str(key) + '.pdf')
  percentage_error_list = []
  for k, j in np.ndenumerate(pred):
    percentage_of_error = abs(((j - lab_np_array_test[k[0]])/(lab_np_array_test[k[0]])) * 100 )
    percentage_error_list.append(percentage_of_error)
  percentage_error_array = np.array(percentage_error_list)
  np.save(savingPath + 'test_error_array' + str(key), percentage_error_array)
  print('For fold {0} and mae validation error {1} the test mean percentage error is: {2} with standard deviation of {3}'.format(key, minimum_value[key], np.mean(percentage_error_array), np.std(percentage_error_array)))
  statSave.write('\t The mae validation error {0} the test mean percentage error is: {1} with standard deviation of {2}'.format(minimum_value[key], np.mean(percentage_error_array), np.std(percentage_error_array)))
  statSave.close()

summ_validation = 0
summ_test = 0
for key in minimum_value.keys():
  summ_validation += minimum_value[key]
  summ_test += test_mae[key]
mean_validation = summ_validation/len(minimum_value.keys())
mean_test = summ_test/len(test_mae.keys())
print('The valuation mean of the mae for {0} folds is: {1}'.format(n_fold,mean_validation))
print('The test mean of the mae for {0} folds  is: {1}'.format(n_fold, mean_test))
statSave = open(savingPath + 'StatisticalAnalysis.txt', 'a+')
statSave.write('The valuation mean of the mae for {0} folds is: {1}'.format(n_fold,mean_validation))
statSave.write('The test mean of the mae for {0} folds  is: {1}'.format(n_fold, mean_test))
statSave.close()
