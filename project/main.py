from reader import DataReader


from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation



if __name__ == "__main__":


    dr = DataReader()
    data = dr.read_set(35)
    dr.plot(data)