from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM

import matplotlib.pyplot as plt

class RespRatePredictor:
    @staticmethod
    def make_network(count_hidden: int = 0, input_shape=()):
        model = Sequential()
        model.add(LSTM(100, input_shape=input_shape))
        for _ in range(count_hidden):
            model.add(Dense(50))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        return model


    @staticmethod
    def fit_network(model, train_X, train_y, test_X, test_y):

        history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()


    # @staticmethod
    # def plot_test(model, test_X):
    #     yhat = model.predict(test_X)
    #     test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    #     # invert scaling for forecast
    #     inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    #     inv_yhat = scaler.inverse_transform(inv_yhat)
    #     inv_yhat = inv_yhat[:, 0]
    #     # invert scaling for actual
    #     test_y = test_y.reshape((len(test_y), 1))
    #     inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    #     inv_y = scaler.inverse_transform(inv_y)
    #     inv_y = inv_y[:, 0]
    #     # calculate RMSE
    #     rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    #     print('Test RMSE: %.3f' % rmse)