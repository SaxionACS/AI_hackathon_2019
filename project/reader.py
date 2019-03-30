import pandas as pnd
import seaborn as sns
import matplotlib.pyplot as plt




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
        print("Loading: {}".format(file_name))
        with open(file_name, "r") as fin:
            data = pnd.read_csv(fin)
            # print(data.head(10))
            data.columns = [col.strip() for col in data.columns]
            return data


    def plot(self, data_set: pnd.DataFrame):
        if "RESP_PREDICTED" in data_set.columns.tolist():
            nrows = 5
        else:
            nrows = 4

        fig, axs = plt.subplots(figsize=[12, nrows*3], nrows=nrows)
        sns.set_context("talk", font_scale=1.0)
        sns.despine()

        sliced = data_set[:15000:5]
        index = 0
        sns.relplot(x='Time [s]', y='RESP', data=sliced, ax=axs[index], kind="line")
        index += 1

        if "RESP_PREDICTED" in data_set.columns.tolist():
            sns.relplot(x='Time [s]', y='RESP_PREDICTED', data=sliced, ax=axs[index], kind="line",
                        color="red")
            index += 1
        sns.relplot(x='Time [s]', y='PLETH', data=sliced, ax=axs[index], kind="line")
        index += 1

        sns.relplot(x='Time [s]', y='AVR', data=sliced, ax=axs[index], kind="line")
        index += 1

        sns.relplot(x='Time [s]', y='V', data=sliced, ax=axs[index], kind="line")

        for i in range(2, nrows+2):
            plt.close(i)

        plt.show()

    def plot_detail(self, data_set: pnd.DataFrame):

        fig, ax = plt.subplots(figsize=[12, 8])
        sns.set_context("talk", font_scale=1.0)
        sns.despine()

        sliced = data_set[0:3750:1]

        sns.relplot(x='Time [s]', y='RESP_PREDICTED', data=sliced, ax=ax, kind="line",
                        color="red")

        sns.relplot(x='Time [s]', y='RESP', data=sliced, ax=ax, kind="line")

        ax.lines[1].set_linestyle("--")

        ax.legend(ax.lines, ["Predicted", "Measured"], loc='upper right', frameon=False)


        plt.close(2)
        plt.close(3)
        plt.show()


