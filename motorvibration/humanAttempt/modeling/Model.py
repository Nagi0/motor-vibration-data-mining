from dataclasses import dataclass
import polars as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
import plotly.express as px


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

    def normalize_data(self, p_df: pd.DataFrame):
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

    training_data = all_motor_operations_df.drop(["file_name", "normal_operation"]).to_pandas()

    scaler, normal_data_scaled = model.normalize_data(
        normal_operation_df.drop(["file_name", "normal_operation"]).to_pandas()
    )
    training_data_scaled = model.apply_normalizer(training_data, scaler)

    print(scaler)
    print(normal_data_scaled.shape)

    kmeans = DBSCAN(eps=30.0)
    y_pred = kmeans.fit_predict(training_data_scaled)
    y_pred[y_pred == 0.0] = 1.0
    y_pred[y_pred == 2.0] = -1.0

    ari = adjusted_rand_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)
    nmi = normalized_mutual_info_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)
    v_measure = v_measure_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    print("V-Measure:", v_measure)

    # Criar DataFrame para facilitar o uso com Plotly
    pca = PCA(n_components=2)
    reduced_data_3d = pca.fit_transform(training_data_scaled)

    plot_data = pd.DataFrame(reduced_data_3d, columns=["PCA1", "PCA2"])
    plot_data["Label"] = all_motor_operations_df["normal_operation"].to_numpy()

    # Plotar com Plotly
    fig = px.scatter(
        plot_data,
        x="PCA1",
        y="PCA2",
        # z="PCA3",
        color="Label",
        title="Anomalias Detectadas com GMM - PCA 3D",
        labels={"Anomaly": "É Anômalo", "Label": "Rótulo Original"},
        color_discrete_map={True: "blue", False: "red"},  # Cores para normal (azul) e anômalo (vermelho)
    )

    # Configurar layout
    fig.update_traces(marker=dict(size=6, opacity=1.0))
    fig.update_layout(
        scene=dict(xaxis_title="PCA1", yaxis_title="PCA2"), template="plotly_dark"
    )  # , zaxis_title="PCA3"))

    # Mostrar o gráfico
    fig.show()

    plot_data["Label"] = y_pred

    # Plotar com Plotly
    fig = px.scatter(
        plot_data,
        x="PCA1",
        y="PCA2",
        # z="PCA3",
        color="Label",
        title="Anomalias Detectadas com GMM - PCA 3D",
        labels={"Anomaly": "É Anômalo", "Label": "Rótulo Agrupamento"},
        color_continuous_scale=px.colors.sequential.Viridis,  # Cores para normal (azul) e anômalo (vermelho)
    )

    # Configurar layout
    fig.update_traces(marker=dict(size=6, opacity=1.0))
    fig.update_layout(
        scene=dict(xaxis_title="PCA1", yaxis_title="PCA2"), template="plotly_dark"
    )  # , zaxis_title="PCA3"))

    # Mostrar o gráfico
    fig.show()
