#%% Import section
from datetime import datetime
import argparse
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
#%% Parsing section
parser = argparse.ArgumentParser()
parser.add_argument('--networks', help='List of network to use between "" ', default='InceptionV3')
parser.add_argument('--network_list_separator', help='Insert the list separator', default=' ')
parser.add_argument('--result_folder', help='Result saving folder', default='Result20802')
parser.add_argument('--data_load_folder', help='Please provide the save data directory', default='Result20802/npdata/')
args = parser.parse_args()

#%% Init section
networks_list = args.networks.split(args.network_list_separator)
save_statistics = args.result_folder + '/Statistics/'
#%% Import data
test_image_set = np.load(args.data_load_folder + 'test_image_set.npy')
test_label_set = np.load(args.data_load_folder + 'test_label_set.npy')
#%%
print('Start model evaluation')
init = datetime.now()
for index, network in enumerate(networks_list):
    save_model_path = args.result_folder + '/' + network
    model = load_model(save_model_path + '/model.h5', compile=False)
    model.compile(optimizer=Adam(lr=0.001, decay=0.00001), loss='mse', metrics=['mae', 'mse'])
    model.load_weights(save_model_path + '/best_weights0.h5')
    statSave = open(save_statistics + 'dEvaluate-Statistics0-Error.txt', 'a+')
    statSave.write('\n')
    # print(model.metrics_names)
    loss, mae, mse = model.evaluate(test_image_set, test_label_set, verbose=0, batch_size=1)
    statSave.write('Network evaluation: ' + network)
    statSave.write('\nTesting set Mean Abs Error: {:5.2f} Kg'.format(mae) + '\n\t Testing Mean Square Error: {:5.2f} kg'.format(mse))
    predicted = model.predict(test_image_set, batch_size=1, verbose=0)
    base_weight = 0
    rweight_total = []
    eweight_total = []
    rweight = []
    eweight = []
    statSave.write('errore Caclcolato con il predict\n')
    for index, value in enumerate(predicted):
        #print(index, value)
        #print(index, test_label_set[index])
        if base_weight == 0:
            base_weight = test_label_set[index]
            rweight.append(test_label_set[index])
            eweight.append(value[0])
        elif base_weight != test_label_set[index]:
            rweight_total.append(rweight)
            eweight_total.append(eweight)
            mean_abs_e = mean_absolute_error(rweight, eweight)
            mean_sqr_e = mean_squared_error(rweight, eweight)
            statSave.write('Weight: {} \t MAE:{} \t MSE:{} \n'.format(test_label_set[index], mean_abs_e, mean_sqr_e))
            rweight = []
            eweight = []
            base_weight = 0
        else:
            rweight.append(test_label_set[index])
            eweight.append(value[0])
    statSave.close()
end = datetime.now()
print('End model evaluation, time required is: {}'.format(end - init))