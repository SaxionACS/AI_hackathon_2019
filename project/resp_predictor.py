from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM, Dropout
import tensorflow.python.keras.regularizers as regularizers
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

class RespRatePredictor:
    @staticmethod
    def make_network(input_shape=()):
        model = Sequential()

        model.add(LSTM(25, activation="tanh",
                           input_shape=input_shape,
                           return_sequences=True))

        model.add(Dropout(0.2))
        model.add(LSTM(50, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='RMSprop')
        return model




    @staticmethod
    def fit_network(model, train_X, train_y, test_X, test_y, batch_size: int = 64):

        history = model.fit(train_X, train_y, epochs=1, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1,
                        shuffle=False)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        return model

    @staticmethod
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    @staticmethod
    def plot_test(model, test_X, test_y, scaler, batch_size: int = 64):
        predict_y = model.predict(test_X, batch_size=batch_size)
        min_ = scaler.min_[1]
        scale_ = scaler.scale_[1]

        predict_y = (predict_y - min_) / scale_
        predict_y = predict_y.flatten()

        test_y = (test_y - min_) / scale_

        up_limit = 3000
        freq = 125.
        time = np.arange(0.0, up_limit/freq, 1.0/freq )

        # predict_y = RespRatePredictor.moving_average(predict_y, 25)
        #
        # print(predict_y)
        # print(test_y)
        # print(time)
        plt.figure(1, figsize=[12, 15])
        plt.subplot(211)
        plt.plot(time, test_y[0:up_limit])
        plt.subplot(212)
        plt.plot(time, predict_y[0:up_limit])
        plt.show()

    @staticmethod
    def plot_test_more(model, test_X, test_y, scaler, batch_size: int = 64):
        predict_y = model.predict(test_X, batch_size=batch_size)
        min_ = scaler.min_[1]
        scale_ = scaler.scale_[1]

        predict_y = (predict_y - min_) / scale_
        predict_y = predict_y.flatten()

        test_y = (test_y - min_) / scale_

        up_limit = 120*125
        freq = 125.
        time = np.arange(0.0, up_limit / freq, 1.0 / freq)

        # predict_y = RespRatePredictor.moving_average(predict_y, 25)
        #
        # print(predict_y)
        # print(test_y)
        # print(time)
        plt.figure(1, figsize=[12, 15])
        plt.subplot(211)
        plt.plot(time[::10], test_y[0:up_limit:10])
        plt.subplot(212)
        plt.plot(time[::10], predict_y[0:up_limit:10])
        plt.show()
