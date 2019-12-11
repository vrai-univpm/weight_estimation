from keras.callbacks import ModelCheckpoint


def saving_callbacks(save_path, fold,save_best_result=True, save_weights_result=True, monitoring_value='val_mae'):
    # Creazione callbacks di salvataggio
    filepath = '/best_weights' + str(fold) + '.h5'  # dynamic name
    checkpoint = ModelCheckpoint(filepath=save_path + filepath, monitor=monitoring_value, verbose=0,
                                 save_best_only=save_best_result, save_weights_only=save_weights_result, mode='min',
                                 period=1)
    return checkpoint