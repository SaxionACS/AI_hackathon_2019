import pandas as pnd

class DataReader:
    @staticmethod
    def read_set(set_number: int) -> pnd.DataFrame:
        """Reads a dataset (signals only) with a given number and return it"""
        file_name = "./data/" + "bidmc_{0:0>2}_Signals.csv".format(set_number)
        with open(file_name, "r") as fin:
            data = pnd.read_csv(fin)
            print(data.head(10))
            return data
