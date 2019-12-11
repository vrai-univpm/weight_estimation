#%%
import pandas as pd
from glob import glob
from skimage.io import imread
import numpy as np
from DifferentCNNTesting.Save.saveStatistics import save_statistics_in_file
#%%
def loading_label(path):
    weights = pd.read_csv(path + 'labels.csv')
    weights = weights['WEIGHT'].values
    return weights

def create_test_dataset(folder_path, save_path, label, train_percentage=0.7, validation_percentage=0.2, test_percentage=0.1):
    from random import shuffle
    # ID list creation:
    id_person_path = sorted(glob(folder_path + 'images/*')) # ricorda di mettere l'* sennò non funziona
    number_of_people = len(id_person_path)
    #print('number_of_people', number_of_people)
    # Defining dimension
    number_of_train_people = int(number_of_people * train_percentage)
    number_of_validation_people = number_of_train_people + int(number_of_people * validation_percentage)
    number_of_test_people = int(number_of_people * test_percentage)
    #print(number_of_test_people)
    # Subdivide IDs in test and execution set:
    shuffle(id_person_path)
    id_person_path_shuffled = id_person_path.copy()
    test_set = id_person_path_shuffled[-number_of_test_people:]  # prendo gli ultimi 11 indici
    cross_set = id_person_path[:-number_of_test_people]  # contiene tutti gli ID per train e validation
    # Test set creation:
    img_array_test = []
    lab_array_test = []
    for id_p in test_set:
        #print(id_p)
        # Obtain ID
        number = int(id_p.split('/')[2])
        # Obtain paths to images
        frames = glob(id_p + '/*')
        # Opening images with skimage
        for image in frames:
            file = imread(image)
            img_array_test.append(file)
            lab_array_test.append(label[number])

    save_statistics_in_file(save_path + '/Test-Statistics.txt', lab_array_test, 'Test')

    # Convertion to numpy array of test set:
    lab_np_array_test = np.array(lab_array_test)
    img_np_array_test = np.array(img_array_test)
    img_np_array_test = np.repeat(img_np_array_test[:, :, :, np.newaxis], 3, axis=3)

    return img_np_array_test, lab_np_array_test, cross_set

def create_train_validation_dataset(save_path, data_path,validation_set, train_set, actual_fold, label, network):
    # Empty Set Creation
    img_array_training = []
    lab_array_training = []
    img_array_validation = []
    lab_array_validation = []
    for id_p in validation_set:
        # Obtain ID
        #number = int(id_p.split('/')[2])
        # Obtain paths to images
        frames = glob(data_path + 'images/' + str(id_p) + '/*')
        # Opening images with skimage
        for image in frames:
            file = imread(image)
            img_array_validation.append(file)
            lab_array_validation.append(label[id_p])

    # Creation of Train Set
    for id_p in train_set:
        # Obtain ID
        #number = int(id_p.split('/')[2])
        # Obtain paths to images
        frames = glob(data_path + 'images/' + str(id_p) + '/*')
        # Opening images with skimage
        for image in frames:
            file = imread(image)
            img_array_training.append(file)
            lab_array_training.append(label[id_p])

    save_statistics_in_file(save_path + '/Test-Validation-Statistics.txt', lab_array_training, '\nTrain - fold: ' + str(actual_fold), network)
    save_statistics_in_file(save_path + '/Test-Validation-Statistics.txt', lab_array_validation, '\nValidation - fold: ' + str(actual_fold), network)

    # Conversion to numpy array:
    img_np_array_training = np.array(img_array_training)
    img_np_array_validation = np.array(img_array_validation)
    lab_np_array_training = np.array(lab_array_training)
    lab_np_array_validation = np.array(lab_array_validation)

    print('img_np_array_training_shape', img_np_array_training.shape)
    print('img_np_array_validation_shape', img_np_array_validation.shape)

    # Duplicazione dei canali
    img_np_array_training = np.repeat(img_np_array_training[:, :, :, np.newaxis], 3, axis=3)
    img_np_array_validation = np.repeat(img_np_array_validation[:, :, :, np.newaxis], 3, axis=3)

    return img_np_array_training, lab_np_array_training, img_np_array_validation, lab_np_array_validation