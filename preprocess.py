import ants
import json
import time

from path import Raw_Data_Dir_Path, Running_File_Dir_Path

def process_mild_depression() -> None:
    """
    ds002748 + ds003007 + Cambridge_Buckner
    """
    ## ds002748
    raw_data_root_dir_path = Raw_Data_Dir_Path.ds002748_dir_path
    saved_dir_path = Running_File_Dir_Path.root_dir / raw_data_root_dir_path.name
    saved_dir_path.mkdir(parents=True, exist_ok=True)
    for sub_dir_path in raw_data_root_dir_path.glob("sub-*"):
        if sub_dir_path.is_dir():
            # dir path
            saved_sub_dir_path = saved_dir_path / sub_dir_path.name
            saved_sub_dir_path.mkdir(parents=True, exist_ok=True)
            # anat
            anat_file_path = list((sub_dir_path / "anat").iterdir())
            assert len(anat_file_path) == 1, f"{sub_dir_path} does not have anat file or more than one"
            anat_file_path = anat_file_path[0]
            # func
            func_file_path = list((sub_dir_path / "func").iterdir())
            assert len(func_file_path) == 1, f"{sub_dir_path} does not have func file or more than one"
            func_file_path = func_file_path[0]
            # 
            

def main():
    process_mild_depression()

if __name__ == "__main__":
    main()