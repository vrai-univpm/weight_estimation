from keras.optimizers import SGD,Adam

def compile_with_adam_optimizer(model, learning_rate, decay=0.00001):
    model.compile(optimizer=Adam(lr=learning_rate, decay=decay), loss='mse', metrics=['mae', 'mse'])
    return model

def compile_with_sgd_optimizer(model, learning_rate, decay=0.00001):
    return model.compile(optimizer=SGD(lr=learning_rate, decay=decay), loss='mse', metrics=['mae', 'mse'])

