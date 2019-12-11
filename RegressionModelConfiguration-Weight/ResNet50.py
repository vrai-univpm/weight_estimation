#%% Import Section
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras import regularizers
from keras.models import load_model

#%%
def create_model(shape):
    base_model = ResNet50(include_top=False, input_tensor=None, input_shape=shape, pooling=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(1, activation='relu')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def model_summary(model):
    model.summary()
    return None

def model_save(model, path):
    model.save(path)
    return None

def model_load(path):
    return load_model(path)
