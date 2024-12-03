from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np
import polars as pl
from scipy import fftpack
import matplotlib.pyplot as plt


@dataclass
class featureEngineering:
    dataset: pl.DataFrame
    frequency: float

    def filter_operations(self, p_df: pl.DataFrame, p_operations_list: list):
        filtered_df_list = []
        for operation in p_operations_list:
            df = p_df.lazy().filter(pl.col("folder").str.contains(f"{operation}")).collect()
            filtered_df_list.append((df, operation))

        return filtered_df_list

    def plot_features(self, p_column: str, p_motor_operations: list):
        filterd_dataset = self.dataset[[p_column, "folder"]]
        filterd_dataset_list = self.filter_operations(filterd_dataset, p_motor_operations)

        legend = []
        plt.figure()
        for operation_df, operation_name in filterd_dataset_list:
            plt.plot(operation_df[p_column])
            legend.append(operation_name)
        plt.legend(legend)
        plt.show()

    def get_spectrum(self, p_y: np.ndarray) -> pl.DataFrame:
        # Number of samplepoints
        length = len(p_y)
        # sample spacing
        sampling_rate = 1.0 / self.frequency
        yf = fftpack.fft(p_y)
        yf = length * np.abs(yf[: length // 2])
        xf = np.linspace(0.0, 1.0 / (2.0 * sampling_rate), length // 2)
        spectrum_df = pl.DataFrame({"xf": xf, "yf": yf})

        return spectrum_df

    def plot_spectrum(self, p_column: str, p_motor_operations: list):
        filterd_dataset = self.dataset[[p_column, "folder"]]
        filterd_dataset_list = self.filter_operations(filterd_dataset, p_motor_operations)

        legend = []
        _, ax = plt.subplots()

        for operation_df, operation_name in filterd_dataset_list:
            spectrum_df = self.get_spectrum(operation_df[p_column].to_numpy())
            spectrum_df = spectrum_df.lazy().filter(pl.col("xf") > 1.0).collect()

            ax.plot(spectrum_df["xf"], spectrum_df["yf"])
            legend.append(operation_name)
        plt.legend(legend)
        plt.show()


if __name__ == "__main__":
    load_dotenv("motorvibration/config/.env")
    dataset = pl.read_csv("motorvibration/Data/motor_vibration_dataset.csv")
    feat_eng = featureEngineering(dataset, 50.0)
    feat_eng.plot_features(
        "overhang_3",
        [
            "motorvibration/Data/normal",
            "motorvibration/Data/imbalance/6g",
            "motorvibration/Data/imbalance/10g",
            "motorvibration/Data/imbalance/15g",
            "motorvibration/Data/imbalance/35g",
        ],
    )
    feat_eng.plot_spectrum(
        "overhang_3",
        [
            "motorvibration/Data/normal",
            "motorvibration/Data/imbalance/6g",
            "motorvibration/Data/imbalance/10g",
            "motorvibration/Data/imbalance/15g",
            "motorvibration/Data/imbalance/35g",
        ],
    )
