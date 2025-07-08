from pathlib import Path
from dataclasses import dataclass

""" Raw Data Paths """
@dataclass(frozen=True)
class Raw_Data_Dir_Path:
    dataset_dir_path = Path("..") / "dataset"
    # mild
    ds002748_dir_path = dataset_dir_path / "ds002748"
    ds003007_dir_path = dataset_dir_path / "ds003007"
    # major
    SRPBS_dir_path    = dataset_dir_path / "SRPBS_OPEN"
    # adolescent
    ds004627_dir_path = dataset_dir_path / "ds004627"
    # health controls
    cambridge_dir_path = dataset_dir_path / "Cambridge_Buckner"
    
""" Brain Atlas Paths """
@dataclass(frozen=True)
class Brain_Atlas_Dir_Path:
    # Brainnetome
    brainnetome_dir_path = Path("Brainnetome_Atlas")
    lut_txt_path = brainnetome_dir_path / "BN_Atlas_246_LUT.txt"
    atlas_1_nii_path = brainnetome_dir_path / "BN_Atlas_246_1mm.nii.gz"
    atlas_2_nii_path = brainnetome_dir_path / "BN_Atlas_246_2mm.nii.gz"
    atlas_3_nii_path = brainnetome_dir_path / "BN_Atlas_246_3mm.nii.gz"
    network_csv_path = brainnetome_dir_path / "subregion_func_network_Yeo_updated.csv"
    assert lut_txt_path.exists(), f"{lut_txt_path} does not exist"
    assert atlas_1_nii_path.exists(), f"{atlas_1_nii_path} does not exist"
    assert atlas_2_nii_path.exists(), f"{atlas_2_nii_path} does not exist"
    assert atlas_3_nii_path.exists(), f"{atlas_3_nii_path} does not exist"
    assert network_csv_path.exists(), f"{network_csv_path} does not exist"
    network_json_path = brainnetome_dir_path / "network.json" # It will be generated in preprocess.py

""" Running File Paths """
@dataclass(frozen=True)
class Running_File_Dir_Path:
    root_dir : Path = Path("..") / "run_files"
    split_dir_path : Path = Path("split")
    split_dir_path.mkdir(parents=True, exist_ok=True)
