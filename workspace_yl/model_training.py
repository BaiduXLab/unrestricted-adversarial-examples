#import bird_or_bicycle
import numpy as np
from keras import applications, layers, models, utils, optimizers, callbacks
from keras import backend as K
import tensorflow as tf
import gc
import glob
from time import time

import pdb

def load_data():
    '''
    data_iter = bird_or_bicycle.get_iterator('train', batch_size=1)
    data_iter = list(data_iter)
    
    X_train = []
    y_train = []
    for data in data_iter:
        X_train.append(data[0][0])
        y_train.append(data[1][0])
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    y_train = utils.to_categorical(y_train, 2).astype('int')
    np.save('X_train', X_train)
    np.save('y_train', y_train)
    '''

    X_data = np.load('X_train.npy')
    y_data = np.load('y_train.npy')

    return X_data, y_data

def get_model(input_shape, num_class):

    input_layer = layers.Input(shape=input_shape)
    #x = layers.Conv2D(64, kernel_size=3, activation='relu')(input_layer)
    #x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = layers.Flatten()(input_layer)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    output_layer = layers.Dense(num_class, activation='softmax')(x)
    model = models.Model(input_layer, output_layer)

    return model

def main():

    X_train, y_train = load_data()

    model = get_model(X_train.shape[1:], y_train.shape[1])

    CMP = callbacks.ModelCheckpoint('./result/temp_model.h5', save_best_only=True, save_weights_only=False)
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=100)
    tensorboard = callbacks.TensorBoard(log_dir="result/logs/{}".format(time()))
    callback_list = [CMP, early_stop, tensorboard]
    optim = optimizers.Adam(0.002)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=300, batch_size=8, validation_split=0.2, verbose=2, callbacks=callback_list)



        







if __name__ == "__main__":
    main()