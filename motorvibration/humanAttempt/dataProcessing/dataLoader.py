import os
from glob import glob
from dataclasses import dataclass
from ast import literal_eval
from dotenv import load_dotenv
from tqdm import tqdm
import polars as pl
from featureEngineering import FeatureEngineering
import matplotlib.pyplot as plt


@dataclass
class DataLoader:
    dataset_path: str

    def list_files(self) -> list:
        csv_files_list = []

        folders_list = os.listdir(self.dataset_path)
        for folder in folders_list:
            folder_path = os.path.join(self.dataset_path, folder)
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

    def load_dataset(self, p_target_feature: str, p_freq_thresh: float, p_top_harmonics: int) -> pl.DataFrame:
        csv_files_list = self.list_files()
        dataframes_list = []

        for file, sub_folder in tqdm(csv_files_list):
            spectrum_feat = {"file_name": file.replace("\\", "/")}
            sub_folder = sub_folder.replace("\\", "/")
            df = pl.read_csv(file, has_header=False, new_columns=literal_eval(os.environ["dataset_columns_name"]))
            feat_eng = FeatureEngineering(df, 50.0)
            spectrum_feat = feat_eng.get_top_harmonics(
                file.replace("\\", "/"), df, p_target_feature, p_freq_thresh, p_top_harmonics
            )
            dataframes_list.append(spectrum_feat)

        full_df = pl.concat(dataframes_list)

        return full_df


if __name__ == "__main__":
    load_dotenv("motorvibration/config/.env")
    data_loader = DataLoader(os.environ["dataset_path"])
    dataset = data_loader.load_dataset(os.environ["target_feature"], p_freq_thresh=5.0, p_top_harmonics=10)
    print(dataset)

    dataset.write_csv(f"{os.environ["dataset_path"]}/motor_vibration_dataset.csv")
