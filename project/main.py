from reader import DataReader


from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation



if __name__ == "__main__":


    dr = DataReader()
    data = dr.read_set(35)
    dr.convert_to_supervised(data, 10, True)