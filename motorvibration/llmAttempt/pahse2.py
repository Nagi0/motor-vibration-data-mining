import os
import polars as pl
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm
from glob import glob


# Step 1: List Files
def list_files(dataset_path):
    csv_files_list = []

    folders_list = os.listdir(dataset_path)
    for folder in folders_list:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            sub_folders_list = os.listdir(folder_path)
            for sub_folder in sub_folders_list:
                sub_folder_path = os.path.join(folder_path, sub_folder)
                if os.path.isdir(sub_folder_path):
                    list_csv = glob(f"{sub_folder_path}/*.csv")
                    for csv_file in list_csv:
                        csv_files_list.append((csv_file, csv_file))
                elif sub_folder_path.endswith(".csv"):
                    csv_files_list.append((sub_folder_path, sub_folder_path))

    return csv_files_list


# Step 2: Load Data
def load_vibration_data_from_list(file_list):
    dataset_columns_name = [
        "tachometer",
        "underhang_1",
        "underhang_2",
        "underhang_3",
        "overhang_1",
        "overhang_2",
        "overhang_3",
        "microphone",
    ]
    data = []
    labels = []

    for file_path, label in tqdm(file_list, desc="Loading files"):
        df = pl.read_csv(file_path, has_header=True, new_columns=dataset_columns_name)
        data.append(df)
        labels.append(label)

    return data, labels


# Step 3: Extract Features
def extract_features(df, sampling_rate):
    signal = df["microphone"].to_numpy()

    # Basic Statistics
    mean = np.mean(signal)
    std_dev = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    kurtosis = pl.Series(signal).kurtosis()
    skewness = pl.Series(signal).skew()

    # Frequency Domain Features
    N = len(signal)
    freq = np.fft.fftfreq(N, d=1 / sampling_rate)
    fft_values = np.abs(fft(signal))
    dominant_freq = freq[np.argmax(fft_values)]
    energy = np.sum(fft_values**2)

    # Time Domain Features
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    zero_crossing_rate = len(zero_crossings) / len(signal)

    return {
        "mean": mean,
        "std_dev": std_dev,
        "min": min_val,
        "max": max_val,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "dominant_frequency": dominant_freq,
        "frequency_energy": energy,
        "zero_crossing_rate": zero_crossing_rate,
    }


# Step 4: Build Feature Matrix
def build_feature_matrix(data, sampling_rate):
    features = []
    for df in tqdm(data, desc="Extracting features"):
        features.append(extract_features(df, sampling_rate))

    return pl.DataFrame(features)


# Step 5: Preprocessing and Clustering
def cluster_data(features):
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Dimensionality Reduction (optional)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(reduced_features)

    return reduced_features, clusters


# Step 6: Visualize Clusters
def visualize_clusters(reduced_features, clusters):
    import matplotlib.pyplot as plt

    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap="viridis")
    plt.colorbar(label="Cluster Label")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("DBSCAN Clustering Results")
    plt.show()


def visualize_labels(file_list, clusters):
    import matplotlib.pyplot as plt

    labels = ["normal" if "normal" in name.lower() else "anomaly" for _, name in file_list]
    color_map = {"normal": "blue", "anomaly": "red"}
    colors = [color_map[label] for label in labels]

    plt.scatter(clusters[:, 0], clusters[:, 1], c=colors, label=labels, alpha=0.6)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Visualization by Labels (Normal vs Anomaly)")
    plt.show()


# Main Pipeline
if __name__ == "__main__":
    # Base path to data
    base_path = "motorvibration/Data"  # Update with the correct path

    # Sampling rate (update as needed)
    sampling_rate = 50000

    # List Files
    file_list = list_files(base_path)
    print("Files found:", file_list)

    # Load Data
    data, labels = load_vibration_data_from_list(file_list)

    # Build Feature Matrix
    features = build_feature_matrix(data, sampling_rate)

    # Clustering
    reduced_features, clusters = cluster_data(features)

    # Visualize Results
    visualize_clusters(reduced_features, clusters)

    # Second Plot: Visualization by Labels
    visualize_labels(file_list, reduced_features)

    # Display cluster assignment
    print("Cluster Assignments:", clusters)
