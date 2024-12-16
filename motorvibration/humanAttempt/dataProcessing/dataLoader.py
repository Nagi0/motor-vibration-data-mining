import os
from glob import glob
from dataclasses import dataclass
from ast import literal_eval
from dotenv import load_dotenv
from tqdm import tqdm
import polars as pl
from featureEngineering import FeatureEngineering


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

    def get_feats_multiple_targets(
        self,
        p_df: pl.DataFrame,
        p_file: str,
        p_feat_eng: FeatureEngineering,
        p_targets_list: list,
        p_freq_thresh: float,
        p_top_harmonics: int,
    ):
        feats_list = []
        for target in p_targets_list:
            wavelet_feat = p_feat_eng.get_wavelet_features(p_df[f"{target}"].to_numpy(), target)
            stats_feat = p_feat_eng.get_stats_features(p_df, target)
            spectrum_feat = p_feat_eng.get_top_harmonics(
                p_file.replace("\\", "/"), p_df, target, p_freq_thresh, p_top_harmonics
            )
            all_feats = pl.concat([spectrum_feat, wavelet_feat, stats_feat], how="horizontal")
            feats_list.append(all_feats)

        return feats_list

    def load_dataset(
        self, p_target_features: str, p_sample_rate: int, p_freq_thresh: float, p_top_harmonics: int
    ) -> pl.DataFrame:
        csv_files_list = self.list_files()
        dataframes_list = []

        for file, sub_folder in tqdm(csv_files_list):
            sub_folder = sub_folder.replace("\\", "/")
            df = pl.read_csv(file, has_header=False, new_columns=literal_eval(os.environ["dataset_columns_name"]))
            feat_eng = FeatureEngineering(df, p_sample_rate)

            feats_list = self.get_feats_multiple_targets(
                df, file, feat_eng, p_target_features, p_freq_thresh, p_top_harmonics
            )

            spectrum_feats_df = pl.concat(feats_list, how="align")
            # spectrum_feats_df = self.get_fftconv_features(feat_eng, df, file)

            dataframes_list.append(spectrum_feats_df)

        full_df = pl.concat(dataframes_list)

        return full_df


if __name__ == "__main__":
    load_dotenv("motorvibration/config/.env")
    data_loader = DataLoader(os.environ["dataset_path"])
    dataset = data_loader.load_dataset(
        literal_eval(os.environ["target_feature"]), p_sample_rate=50000, p_freq_thresh=1000.0, p_top_harmonics=500
    )
    print(dataset)

    dataset.write_csv(f"{os.environ["dataset_path"]}/motor_vibration_dataset.csv")
