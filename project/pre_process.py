from sklearn.preprocessing import MinMaxScaler
import pandas as pnd


class Preprocess:

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

    @staticmethod
    def resample(data_set: pnd.DataFrame, resample_interval: int = 1) -> pnd.DataFrame:
        return data_set[::resample_interval]

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
    def prepare(data_set: pnd.DataFrame, test_size: float=0.2):
        # remove time
        data = data_set.drop("Time [s]", axis=1)
        cut = int(len(data.index) * (1. - test_size))
        training_set, test_set = data[:cut], data[cut:]
        return training_set, test_set