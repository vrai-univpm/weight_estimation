#%% Import Section
import os
from datetime import datetime
import argparse
import importlib
import tensorflow as tf
from datetime import datetime
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#%% General Parameters Section
now = datetime.now()

epochs = 200
batch_size = 32
shape = (150, 150, 3)
learning_rate = 0.001
decay = 0.00001
n_fold = 3

#%% model preparation
parser = argparse.ArgumentParser()
parser.add_argument('--networks', help='List of network to use between "" ', default='Inception-V3')
parser.add_argument('--network_list_separator', help='Insert the list separator', default=' ')
parser.add_argument('--batch_size', help='Batch size for the exectuion', default=batch_size)
parser.add_argument('--pretrained', help='Boolean choosing for pretrained or not network', default=False)
parser.add_argument('--optimizer', help='Choose optimizer', default='adam')
parser.add_argument('--learning_rate', help='Choose learning rate', default=learning_rate)
parser.add_argument('--data_folder', help='Folder with all dataset', required=True)
parser.add_argument('--result_folder', help='Result saving folder', default='Result')
parser.add_argument('--epochs', help='Number of hepocs', default=epochs)
parser.add_argument('--models', help='File that contain the model configuration in python', default='DefaultModelConfiguration')
parser.add_argument('--train_percentage', help='Please define the percentage of data used for testing', default='0.7')
parser.add_argument('--validation_percentage', help='Please define the percentage of data used for validation', default='0.2')
parser.add_argument('--test_percentage', help='Please define the percentage of data used for validation', default='0.1')
#parser.add_argument('')
args = parser.parse_args()
# For dinamical import importlib.import_module('.inceptionV3', args.models)
#TODO aggiungere il parsing all'inizio andando a fare casting a priori
#%% Model generator
print('Start model generator')
init = datetime.now()
compile_model = importlib.import_module('.create_optimizer', args.models)
networks_list = args.networks.split(args.network_list_separator)
if not os.path.exists(args.result_folder):
    os.mkdir(args.result_folder)
save_statistics = args.result_folder + '/Statistics/'
if not os.path.exists(save_statistics):
    os.mkdir(save_statistics)
generators = []
for index, network in enumerate(networks_list):
    generators.append(importlib.import_module('.'+network, args.models))
    model = generators[index].create_model(shape) # creao il modello senza pesi
    save_model_path = args.result_folder + '/' + network
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    model = compile_model.compile_with_adam_optimizer(model, args.learning_rate) # ricorda che il modello va compilato prima di salvarlo
    generators[index].model_save(model, save_model_path + '/model.h5')
end = datetime.now()
print('End model generator, time required is: {}'.format(end - init))
#%% Training
print('Start model train')
init = datetime.now()
data_augmentation = importlib.import_module('.data_augmentation', args.models)
dataset_augmentation = data_augmentation.data_augmentation()
call_creation = importlib.import_module('.callbacks', args.models)
data_creation = importlib.import_module('.data_creation', args.models)
labels = data_creation.loading_label(args.data_folder)
test_image_set, test_label_set, cross_set = data_creation.create_test_dataset(args.data_folder, save_statistics,labels)
networks_history = {}
for index, network in enumerate(networks_list):
    print(network)
    history = []
    for fold in range(n_fold):
        print(fold)
        train_set, train_label, validation_set, validation_label = data_creation.create_train_validation_dataset(save_statistics, cross_set, n_fold, fold, labels)
        save_model_path = args.result_folder + '/' + network
        model_weight_save_callback = call_creation.saving_callbacks(save_model_path, fold)
        model = generators[index].model_load(save_model_path + '/model.h5')
        # Starting Training
        history.append(model.fit_generator(
            dataset_augmentation.flow(train_set, train_label, batch_size=int(args.batch_size)),
            steps_per_epoch=len(train_set) / batch_size, epochs=int(args.epochs),
            validation_data=(validation_set, validation_label), validation_steps=len(validation_set) / batch_size,
            callbacks=[model_weight_save_callback]))
    networks_history[network] = history

end = datetime.now()
print('End models train, time required is: {}'.format(end - init))
#%% Evaluate
print('Start model evaluation')
init = datetime.now()
for index, network in enumerate(networks_list):
    save_model_path = args.result_folder + '/' + network
    model = generators[index].model_load(save_model_path + '/model.h5')
    model.load_weights(save_model_path + '/best_weights.h5')
    statSave = open(save_statistics + 'Evaluate-Statistics.txt', 'a+')
    statSave.write('\n')
    # print(model.metrics_names)
    loss, mae, mse = model.evaluate(test_image_set, test_label_set, verbose=0)
    statSave.write('Network evaluation: ' + network)
    statSave.write('\nTesting set Mean Abs Error: {:5.2f} Kg'.format(mae) + '\n\t Testing Mean Square Error: {:5.2f} kg'.format(mse))
    statSave.close()
end = datetime.now()
print('End model evaluation, time required is: {}'.format(end - init))
#%%
