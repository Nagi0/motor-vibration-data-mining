import os
from ast import literal_eval
from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np
import polars as pl
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)
import plotly.express as px


@dataclass
class Model:
    dataset_path: str

    def filter_motor_operation(self, p_df: pl.DataFrame, p_name: str) -> pl.DataFrame:
        return p_df.lazy().filter(pl.col("label") == p_name).collect()

    def filter_not_motor_operation(self, p_df: pl.DataFrame, p_name: str) -> pl.DataFrame:
        return p_df.lazy().filter(pl.col("label") != p_name).collect()

    def create_class_labels(self, p_df: pl.DataFrame, p_labels_name_list: list) -> np.array:
        labeled_df_list = []

        for label_name in p_labels_name_list:
            df = p_df.lazy().filter(pl.col("file_name").str.contains(f"{label_name}")).collect()
            df = df.with_columns(pl.lit(label_name).alias("label"))
            labeled_df_list.append(df)

        return pl.concat(labeled_df_list)

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
    load_dotenv("motorvibration/config/.env")
    model = Model("motorvibration/Data/motor_vibration_dataset.csv")
    dataset = model.load_dataset()

    dataset = model.create_class_labels(dataset, literal_eval(os.environ["labels_list"]))

    normal_operation_df = model.filter_motor_operation(dataset, "normal")
    normal_operation_df = normal_operation_df.with_columns(pl.lit(True).alias("normal_operation"))

    not_normal_operation_df = model.filter_not_motor_operation(dataset, "normal")
    not_normal_operation_df = not_normal_operation_df.with_columns(pl.lit(False).alias("normal_operation"))

    all_motor_operations_df = pl.concat([normal_operation_df, not_normal_operation_df])
    print(all_motor_operations_df)

    harmonic = 1
    x_column = f"overhang_3_harmonic_{harmonic}_frequency"
    y_column = f"overhang_3_harmonic_{harmonic}"
    fig = px.scatter(
        all_motor_operations_df,
        x=x_column,
        y=y_column,
        color="normal_operation",
        hover_data=["file_name"],
        title=f"Scatter Plot Interativo: {x_column} vs {y_column}",
        labels={"normal_operation": "Operação Normal"},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title="Operação",
    )
    fig.show()

    training_data = all_motor_operations_df.drop(["file_name", "label", "normal_operation"]).to_pandas()

    scaler, normal_data_scaled = model.normalize_data(
        normal_operation_df.drop(["file_name", "label", "normal_operation"]).to_pandas()
    )
    training_data_scaled = model.apply_normalizer(training_data, scaler)

    reducer = umap.UMAP(metric="manhattan", random_state=42)
    embeddings = reducer.fit_transform(
        X=training_data_scaled, y=all_motor_operations_df["normal_operation"].to_numpy()
    )

    plot_data = pd.DataFrame(embeddings, columns=["Component1", "Component2"])
    plot_data["Label"] = all_motor_operations_df["label"].to_numpy()

    fig = px.scatter(
        plot_data,
        x="Component1",
        y="Component2",
        color="Label",
        title="Dados de Operações do Motor Rotuladas",
        labels={"Anomaly": "É Anômalo", "Label": "Rótulo Original"},
    )

    fig.update_traces(marker=dict(size=6, opacity=1.0))
    fig.update_layout(scene=dict(xaxis_title="PCA1", yaxis_title="PCA2"))
    fig.show()

    dbscan = DBSCAN(eps=30.0, min_samples=10)
    y_pred = dbscan.fit_predict(training_data_scaled)

    ari = adjusted_rand_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)
    nmi = normalized_mutual_info_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)
    v_measure = v_measure_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)

    print("Internal Metrics: \n")
    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    print("V-Measure:", v_measure)

    print("External Metrics: \n")
    davies_bouding = davies_bouldin_score(training_data_scaled, y_pred)
    silhouette = silhouette_score(training_data_scaled, y_pred)
    calinski = calinski_harabasz_score(training_data_scaled, y_pred)
    print(f"davies_bouldin_score: {davies_bouding}")
    print(f"silhouette_score: {silhouette}")
    print(f"calinski_harabasz_score: {calinski}")

    plot_data = pd.DataFrame(embeddings, columns=["Component1", "Component2"])
    plot_data["Label"] = all_motor_operations_df["normal_operation"].to_numpy()
    fig = px.scatter(
        plot_data,
        x="Component1",
        y="Component2",
        color="Label",
        title="Dados de Operações do Motor com Rótulos Binários",
        labels={"Label": "Rótulo Original"},
    )
    fig.update_traces(marker=dict(size=6, opacity=1.0))
    fig.update_layout(scene=dict(xaxis_title="Component1", yaxis_title="Component2"))
    fig.show()

    plot_data["Label"] = list(map(str, y_pred))
    fig = px.scatter(
        plot_data,
        x="Component1",
        y="Component2",
        color="Label",
        title="DBSCAN - Agrupamento Realizado para Detecção de Anomalias",
        labels={"Anomaly": "É Anômalo", "Label": "Rótulo Agrupamento"},
    )
    fig.update_traces(marker=dict(size=6, opacity=1.0))
    fig.update_layout(scene=dict(xaxis_title="PCA1", yaxis_title="PCA2"))
    fig.show()

    y_pred = dbscan.fit_predict(training_data_scaled)
    y_pred[y_pred == 0.0] = 1.0

    ari = adjusted_rand_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)
    nmi = normalized_mutual_info_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)
    v_measure = v_measure_score(all_motor_operations_df["normal_operation"].to_numpy(), y_pred)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    print("V-Measure:", v_measure)

    plot_data["Label"] = list(map(str, y_pred))
    fig = px.scatter(
        plot_data,
        x="Component1",
        y="Component2",
        color="Label",
        title="DBSCAN - Agrupamento Realizado para Detecção de Anomalias (Junção dos Cluster de Operação Normal)",
        labels={"Anomaly": "É Anômalo", "Label": "Rótulo Agrupamento"},
    )

    fig.update_traces(marker=dict(size=6, opacity=1.0))
    fig.update_layout(scene=dict(xaxis_title="Component1", yaxis_title="Component2"))

    fig.show()
