from keras.preprocessing.image import ImageDataGenerator

def data_augmentation():
    # Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    return datagen