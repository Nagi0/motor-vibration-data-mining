from dataclasses import dataclass
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import seaborn as sns
import plotly.express as px
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

    # Escolher os harmônicos e frequências para o scatter plot
    harmonic = 1
    x_column = f"overhang_3_harmonic_{harmonic}_frequency"
    y_column = f"overhang_3_harmonic_{harmonic}"

    # Criar scatter plot interativo
    fig = px.scatter(
        all_motor_operations_df,
        x=x_column,
        y=y_column,
        color="normal_operation",
        hover_data=["file_name"],
        title=f"Scatter Plot Interativo: {x_column} vs {y_column}",
        labels={"normal_operation": "Operação Normal"},
    )

    # Configurar layout do gráfico
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title="Operação",
    )

    # Exibir o gráfico
    fig.show()

    training_data = normal_operation_df.drop(["file_name", "normal_operation"]).to_pandas()
    print(training_data)
    test_data = all_motor_operations_df.drop(["file_name", "normal_operation"]).to_pandas()

    scaler, normal_data_scaled = model.normalize_data(training_data)
    all_data_scaled = model.apply_normalizer(test_data, scaler)

    print(scaler)
    print(normal_data_scaled.shape)

    gmm = BayesianGaussianMixture(
        n_components=2, covariance_type="full", init_params="k-means++", max_iter=500
    )  # Ou use BayesianGaussianMixture
    gmm = gmm.fit(normal_data_scaled)

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

    pca = PCA(n_components=3)
    reduced_data_3d = pca.fit_transform(all_data_scaled)

    # Criar DataFrame para facilitar o uso com Plotly
    plot_data = pd.DataFrame(reduced_data_3d, columns=["PCA1", "PCA2", "PCA3"])
    plot_data["Label"] = original_labels  # Adicionar rótulos originais

    # Plotar com Plotly
    fig = px.scatter_3d(
        plot_data,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Label",
        title="Anomalias Detectadas com GMM - PCA 3D",
        labels={"Anomaly": "É Anômalo", "Label": "Rótulo Original"},
        color_discrete_map={True: "blue", False: "red"},  # Cores para normal (azul) e anômalo (vermelho)
    )

    # Configurar layout
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"))

    # Mostrar o gráfico
    fig.show()
