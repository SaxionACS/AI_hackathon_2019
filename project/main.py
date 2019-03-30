from reader import DataReader
from pre_process import Preprocess
from resp_predictor import RespRatePredictor
from tensorflow.python.keras.models import load_model
import pandas as pnd

from typing import List, AnyStr, Dict
import os

class TextUI:
    __prefix_data = "./data/"
    __data_indices = [2, 11, 18, 23, 42, 44, 46, 50, 52]

    __files: List[AnyStr]

    def __init__(self):
        self.dr = DataReader()

    def find_data_files(self):
        self.__files = []
        for file in os.listdir(self.__prefix_data):
            if file.endswith(".csv") and "Signals" in file:
                if any ((("{0:0>2}".format(i) in file) for i in self.__data_indices)):
                        self.__files.append(file)

        self.__files.sort()

    def run(self):
        model = None
        scaler = None
        while True:
            self.find_data_files()

            print("Enter:\nt - to train the model, \ne - to test a trained model, \nl - to load a pre-trained model\nq - to quit")
            print("IMPORTANT: Always train or load a model before testing it!\n")
            choice = input("Your choice: ")
            if choice is "l":
                model = load_model("model.h5")
            elif choice is "t" or choice is "e":
                print("Data files:\n")
                for i, file in enumerate(self.__files):
                    print("{} - {}".format(i+1, file))

                if choice is "t":
                    number = input("\nPlease select the file to train the model on: ")
                else:
                    number = input("\nPlease select the file to test the model on: ")

                index = int(number)
                index -= 1

                if 0 <= index < len(self.__files):
                    data = self.dr.read_set(self.__data_indices[index])

                    pp = Preprocess()
                    data, scaler = pp.clean_up(data)
                    data = pp.convert_to_supervised(data, sample_shift=0)
                    if choice is "t":
                        train, test = pp.prepare_sets(data, 0.2)
                        train_X, train_y = pp.make_input_output(train, remove_resp_from_input=True)
                        test_X, test_y = pp.make_input_output(test, remove_resp_from_input=True)
                        trainer = RespRatePredictor()
                        self.dr.plot(data)
                        model = trainer.make_network(input_shape=(train_X.shape[1], train_X.shape[2]))
                        model = trainer.fit_network(model, train_X, train_y, test_X, test_y)
                        model.save("model_{0:0>2}.h5".format(self.__data_indices[index - 1]))
                    else:
                        all_X, all_y = pp.make_input_output(data.drop("Time [s]", axis=1),
                                                            remove_resp_from_input=True)
                        predict_y = model.predict(all_X, batch_size=640)
                        # min_ = scaler.min_[1]
                        # scale_ = scaler.scale_[1]

                        # predict_y = (predict_y - min_) / scale_
                        predicted = pnd.DataFrame({"RESP_PREDICTED": predict_y.flatten()})

                        fused = pnd.concat([data, predicted], axis=1)
                        self.dr.plot(fused)
                        self.dr.plot_detail(fused)
                else:
                    continue
            else:
                break

if __name__ == "__main__":
    ui = TextUI()
    ui.run()








