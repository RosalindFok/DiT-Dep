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
    # check 
    assert SRPBS_dir_path.exists(), f"{SRPBS_dir_path} does not exist"
    assert ds002748_dir_path.exists(), f"{ds002748_dir_path} does not exist"
    assert ds003007_dir_path.exists(), f"{ds003007_dir_path} does not exist"
    assert ds004627_dir_path.exists(), f"{ds004627_dir_path} does not exist"
    assert cambridge_dir_path.exists(), f"{cambridge_dir_path} does not exist"
    
""" Brain Atlas Paths """
@dataclass(frozen=True)
class Brain_Atlas_Dir_Path:
    # Brainnetome
    brainnetome_dir_path = Path("Brainnetome_Atlas")
    assert brainnetome_dir_path.exists(), f"{brainnetome_dir_path} does not exist"

""" Running File Paths """
@dataclass(frozen=True)
class Running_File_Dir_Path:
    root_dir = Path("..") / "run_files"
    