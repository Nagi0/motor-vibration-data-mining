import os
from glob import glob
from dataclasses import dataclass
from ast import literal_eval
from dotenv import load_dotenv
from tqdm import tqdm
import polars as pl


@dataclass
class dataLoader:
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
                            csv_files_list.append((csv_file, sub_folder_path))
                    elif sub_folder_path.endswith(".csv"):
                        csv_files_list.append((sub_folder_path, sub_folder_path))

        return csv_files_list

    def load_dataset(self) -> pl.DataFrame:
        csv_files_list = self.list_files()
        dataframes_list = []

        for file, sub_folder in tqdm(csv_files_list):
            sub_folder = sub_folder.replace("\\", "/")
            df = pl.read_csv(file, has_header=False, new_columns=literal_eval(os.environ["dataset_columns_name"]))
            df = df.lazy().with_columns(pl.lit(sub_folder).alias("folder")).collect()
            dataframes_list.append(df)

        full_df = pl.concat(dataframes_list)

        return full_df


if __name__ == "__main__":
    load_dotenv("motorvibration/config/.env")
    data_loader = dataLoader(os.environ["dataset_path"])
    dataset = data_loader.load_dataset()
    print(dataset)

    dataset.write_csv(f"{os.environ["dataset_path"]}/motor_vibration_dataset.csv")
