from reader import DataReader
from pre_process import Preprocess
from resp_predictor import RespRatePredictor



if __name__ == "__main__":


    dr = DataReader()
    data = dr.read_set(25)
    # dr.plot(data)
    pp = Preprocess()
    data = pp.clean_up(data)
    data = pp.convert_to_supervised(data, sample_shift=10)
    train, test = pp.prepare_sets(data, 0.2)
    print(train.head(10))
    train_X, train_y = pp.make_input_output(train, remove_resp_from_input=False)
    test_X, test_y = pp.make_input_output(test, remove_resp_from_input=False)

    print(train_X[:10, :])


    model = RespRatePredictor()
    #
    network = model.make_network(count_hidden=1, input_shape=(train_X.shape[1], train_X.shape[2]))
    #
    model.fit_network(network, train_X, train_y, test_X, test_y)
