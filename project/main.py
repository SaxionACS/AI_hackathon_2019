from reader import DataReader
from pre_process import Preprocess

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation



if __name__ == "__main__":


    dr = DataReader()
    data = dr.read_set(25)
    # dr.plot(data)
    pp = Preprocess()
    data = pp.clean_up(data)
    data = pp.convert_to_supervised(data, sample_shift=10)
    train, test = pp.prepare_sets(data, 0.2)
    print(train.head(10))
    train_X, train_y = pp.make_input_output(data)

    print(train_X[:10])
    print(train_y[:10])
    # dr.convert_to_supervised(data, 10, True)