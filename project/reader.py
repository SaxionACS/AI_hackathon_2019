import pandas as pnd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler



class DataReader:
    @staticmethod
    def read_set(set_number: int) -> pnd.DataFrame:
        """Reads a dataset (signals only) with a given number and returns it
        Columns are:
        RESP - respiration waveform
        PLETH - fingertip plethysmograph (PPG)
        V,
        AVR,
        II
        """
        file_name = "./data/" + "bidmc_{0:0>2}_Signals.csv".format(set_number)
        with open(file_name, "r") as fin:
            data = pnd.read_csv(fin)
            # print(data.head(10))
            data.columns = [col.strip() for col in data.columns]
            return data

    @staticmethod
    def convert_to_supervised(data_set: pnd.DataFrame, sample_shift: int = 10, drop_nan: bool = True) -> pnd.DataFrame:
        """Adds a column shifted by sample_shift samples"""
        data = pnd.DataFrame(data_set)
        data["Y"] = data_set["RESP"].shift(sample_shift)
        if drop_nan:
            # remove NaN
            data.dropna(inplace=True)
        return data

    @staticmethod
    def clean_up(data_set: pnd.DataFrame) -> pnd.DataFrame:
        # remove NaN
        data = data_set.dropna()
        min_max_scaler = MinMaxScaler()
        data = pnd.DataFrame(min_max_scaler.fit_transform(data.values), columns=data.columns, index=data.index)
        if len(data.index) == len(data_set.index):
            data["Time [s]"] = data_set["Time [s]"]
        else:
            temp = data_set.dropna()
            data["Time [s]"] = temp["Time [s]"]
        return data



    def plot(self, data_set: pnd.DataFrame):
        fig, axs = plt.subplots(figsize=[12, 15], nrows=5)
        sns.set_context("talk", font_scale=1.0)
        sns.despine()
        print(data_set.columns.to_list())
        sliced = data_set[0:7500:5]
        sns.relplot(x='Time [s]', y='RESP', data=sliced, ax=axs[0], kind="line")
        sns.relplot(x='Time [s]', y='PLETH', data=sliced, ax=axs[1], kind="line")
        sns.relplot(x='Time [s]', y='II', data=sliced, ax=axs[2], kind="line")
        sns.relplot(x='Time [s]', y='AVR', data=sliced, ax=axs[3], kind="line")
        sns.relplot(x='Time [s]', y='V', data=sliced, ax=axs[4], kind="line")

        plt.show()


