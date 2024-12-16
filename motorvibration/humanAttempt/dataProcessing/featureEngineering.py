import os
from dataclasses import dataclass
from ast import literal_eval
from dotenv import load_dotenv
import numpy as np
import polars as pl
from scipy import fftpack
import pywt
import matplotlib.pyplot as plt


@dataclass
class FeatureEngineering:
    dataset: pl.DataFrame
    frequency_hz: float

    def filter_operations(self, p_df: pl.DataFrame, p_operations_list: list):
        filtered_df_list = []
        for operation in p_operations_list:
            df = p_df.lazy().filter(pl.col("file_name").str.contains(f"{operation}")).collect()
            filtered_df_list.append((df, operation))

        return filtered_df_list

    def plot_features(self, p_column: str, p_motor_operations: list):
        filterd_dataset = self.dataset[[p_column, "file_name"]]
        filterd_dataset_list = self.filter_operations(filterd_dataset, p_motor_operations)

        legend = []
        plt.figure()
        for operation_df, operation_name in filterd_dataset_list:
            plt.plot(operation_df[p_column])
            legend.append(operation_name)
        plt.legend(legend)
        plt.show()

    def get_wavelet_features(self, p_signal: np.ndarray, p_target: str, p_wavelet="db2", p_levels=4):
        # Decompor o sinal em níveis usando DWT
        coeffs = pywt.wavedec(p_signal, p_wavelet, level=p_levels)

        features = {}

        # Para cada nível de decomposição (aproximação e detalhes)
        for i, coeff in enumerate(coeffs):
            # Estatísticas dos coeficientes
            features[f"{p_target}_level_{i}_mean"] = np.mean(coeff)
            features[f"{p_target}_level_{i}_std"] = np.std(coeff)
            features[f"{p_target}_level_{i}_energy"] = np.sum(coeff**2)
            features[f"{p_target}_level_{i}_kurtosis"] = np.mean((coeff - np.mean(coeff)) ** 4) / (np.std(coeff) ** 4)
            features[f"{p_target}_level_{i}_skewness"] = np.mean((coeff - np.mean(coeff)) ** 3) / (np.std(coeff) ** 3)

        return pl.DataFrame(features)

    def get_spectrum(self, p_y: np.ndarray) -> pl.DataFrame:
        yf = fftpack.fft(p_y)
        xf = fftpack.fftfreq(len(p_y), 1 / self.frequency_hz)
        spectrum_df = pl.DataFrame({"xf": xf, "yf": np.abs(yf)})

        return spectrum_df

    def get_greatests_harmonics(self, p_df: pl.DataFrame, top_n: int, p_min_freq: float) -> pl.DataFrame:
        p_df = p_df.lazy().filter(pl.col("xf") > p_min_freq).collect()
        return p_df.sort(by="yf", descending=True).head(top_n)

    def get_top_harmonics(
        self, p_name: str, p_df: pl.DataFrame, p_target_feature: str, p_freq_thresh: float, p_top_harmonics: int
    ) -> pl.DataFrame:
        spectrum_feat = {"file_name": p_name}
        spectrum_df = self.get_spectrum(p_df[p_target_feature].to_numpy())
        spectrum_df = self.get_greatests_harmonics(spectrum_df, p_top_harmonics, p_freq_thresh)

        for idx in range(p_top_harmonics):
            spectrum_feat[f"{p_target_feature}_harmonic_{idx+1}_frequency"] = spectrum_df.row(idx, named=True)["xf"]
            spectrum_feat[f"{p_target_feature}_harmonic_{idx+1}"] = spectrum_df.row(idx, named=True)["yf"]

        return pl.DataFrame(spectrum_feat)

    def get_stats_features(self, p_df: pl.DataFrame, p_target_feature: str) -> pl.DataFrame:
        feat_dict = {}

        df = p_df.with_columns(
            [
                pl.col(p_target_feature).min().alias("min"),
                pl.col(p_target_feature).max().alias("max"),
                pl.col(p_target_feature).mean().alias("mean"),
                pl.col(p_target_feature).var().alias("var"),
                pl.col(p_target_feature).kurtosis().alias("kurtosis"),
                pl.col(p_target_feature).skew().alias("skewness"),
                pl.col(p_target_feature).abs().entropy(base=2).alias("entropy"),
            ]
        )
        feat_dict[f"{p_target_feature}_min"] = df["min"][0]
        feat_dict[f"{p_target_feature}_max"] = df["max"][0]
        feat_dict[f"{p_target_feature}_mean"] = df["mean"][0]
        feat_dict[f"{p_target_feature}_var"] = df["var"][0]
        feat_dict[f"{p_target_feature}_kurtosis"] = df["kurtosis"][0]
        feat_dict[f"{p_target_feature}_skewness"] = df["skewness"][0]
        feat_dict[f"{p_target_feature}_entropy"] = df["entropy"][0]

        return pl.DataFrame(feat_dict)

    def plot_spectrum(self, p_column: str, p_motor_operations: list, p_freq_thresh: float):
        filterd_dataset = self.dataset[[p_column, "file_name"]]
        filterd_dataset_list = self.filter_operations(filterd_dataset, p_motor_operations)

        legend = []
        _, ax = plt.subplots()

        for operation_df, operation_name in filterd_dataset_list:
            spectrum_df = self.get_spectrum(operation_df[p_column].to_numpy())
            spectrum_df = spectrum_df.lazy().filter(pl.col("xf") > p_freq_thresh).collect()

            ax.plot(spectrum_df["xf"], spectrum_df["yf"])
            legend.append(operation_name)
        plt.legend(legend)
        plt.show()


if __name__ == "__main__":

    load_dotenv("motorvibration/config/.env")
    files_list = [
        "motorvibration/Data/normal/12.288.csv",
        "motorvibration/Data/normal/13.1072.csv",
        "motorvibration/Data/normal/14.336.csv",
    ]

    datasets_list = []
    for file in files_list:
        dataset = pl.read_csv(file, new_columns=literal_eval(os.environ["dataset_columns_name"]))
        dataset = dataset.with_columns(pl.lit(file).alias("file_name"))
        datasets_list.append(dataset)

    dataset = pl.concat(datasets_list)
    print(dataset)
    freq = 50000
    feat_eng = FeatureEngineering(dataset, 50)
    feat_eng.plot_spectrum(
        "overhang_3",
        [
            "motorvibration/Data/normal/12.288.csv",
            "motorvibration/Data/normal/13.1072.csv",
            "motorvibration/Data/normal/14.336.csv",
        ],
        15.0,
    )
