from dataclasses import dataclass
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class Model:
    dataset_path: str

    def filter_motor_operation(self, p_df: pl.DataFrame, p_name: str) -> pl.DataFrame:
        return p_df.lazy().filter(pl.col("file_name").str.contains(f"{p_name}")).collect()

    def filter_not_motor_operation(self, p_df: pl.DataFrame, p_name: str) -> pl.DataFrame:
        return p_df.lazy().filter(~pl.col("file_name").str.contains(f"{p_name}")).collect()

    def load_dataset(self):
        return pl.read_csv(self.dataset_path)

    def get_min_max_normalizer(self, p_df: pd.DataFrame) -> MinMaxScaler:
        return MinMaxScaler().fit(p_df)

    def normalize_data(self, p_df):
        scaler = self.get_min_max_normalizer(p_df)
        return scaler, scaler.transform(p_df)

    def apply_normalizer(self, p_df: pd.DataFrame, p_scaler: MinMaxScaler):
        return p_scaler.transform(p_df)


if __name__ == "__main__":
    model = Model("motorvibration/Data/motor_vibration_dataset.csv")
    dataset = model.load_dataset()

    normal_operation_df = model.filter_motor_operation(dataset, "normal")
    normal_operation_df = normal_operation_df.with_columns(pl.lit(True).alias("normal_operation"))

    not_normal_operation_df = model.filter_not_motor_operation(dataset, "normal")
    not_normal_operation_df = not_normal_operation_df.with_columns(pl.lit(False).alias("normal_operation"))

    all_motor_operations_df = pl.concat([normal_operation_df, not_normal_operation_df])
    print(all_motor_operations_df)

    harmonic = 1
    fg = sns.FacetGrid(data=all_motor_operations_df, hue="normal_operation", aspect=1.61)
    fg.map(plt.scatter, f"harmonic_{harmonic}_frequency", f"harmonic_{harmonic}").add_legend()
    plt.show()

    training_data = normal_operation_df.drop(["file_name", "normal_operation"]).to_pandas()
    print(training_data)
    test_data = all_motor_operations_df.drop(["file_name", "normal_operation"]).to_pandas()

    scaler, normal_data_scaled = model.normalize_data(training_data)
    all_data_scaled = model.apply_normalizer(test_data, scaler)

    print(scaler)
    print(normal_data_scaled.shape)

    gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=42)  # Ou use BayesianGaussianMixture
    gmm.fit(normal_data_scaled)

    # Calcular a probabilidade de cada ponto pertencer a cada gaussiana
    probabilities = gmm.predict_proba(all_data_scaled)

    # Identificar a probabilidade máxima para cada ponto
    max_probabilities = np.max(probabilities, axis=1)

    # Definir limiar para anomalias (por exemplo, probabilidade < 0.05 é anômala)
    anomaly_threshold = 0.05
    is_anomaly = max_probabilities < anomaly_threshold

    # Adicionar rótulos ao DataFrame
    dataset = dataset.with_columns(
        pl.Series(name="max_probability", values=max_probabilities), pl.Series(name="is_anomaly", values=is_anomaly)
    )

    # Visualizar a distribuição das probabilidades
    plt.figure(figsize=(10, 6))
    sns.histplot(max_probabilities, bins=50, kde=True)
    plt.axvline(anomaly_threshold, color="red", linestyle="--", label="Anomaly Threshold")
    plt.title("Distribuição das Probabilidades Máximas")
    plt.xlabel("Probabilidade Máxima")
    plt.ylabel("Frequência")
    plt.legend()
    plt.show()

    # Relatório de classificação
    original_labels = all_motor_operations_df["normal_operation"].to_pandas().astype(int)
    predicted_labels = (~is_anomaly).astype(int)

    print("Relatório de Classificação:")
    print(classification_report(original_labels, predicted_labels))

    # Visualizar em PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(normal_data_scaled)
    reduced_data = pca.transform(all_data_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=original_labels, palette="coolwarm", legend="full")
    plt.title("Anomalias Detectadas com GMM")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="É Anômalo")
    plt.show()
