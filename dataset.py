import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from torch.utils.data import Dataset

from config import Experiment_Config
from utils import to_tensor, read_json
from path import Running_File_Dir_Path

# XAI

# dataclass is not used because dict[tensor] is not supported by PyTorch
Dataset_Returns = namedtuple(
    typename="Dataset_Returns", 
    field_names=[
        # data derived from rs-fMRI
        Experiment_Config.TS, Experiment_Config.FC, 
        # auxiliary information
        Experiment_Config.A, Experiment_Config.G, Experiment_Config.H, 
        # sub_id and target
        Experiment_Config.I, Experiment_Config.T
])

def get_leaf_values(d : dict) -> list:
    values = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(get_leaf_values(d=v))
        else:
            values.extend(list(v))
    return values

class Depression_Dataset(Dataset):
    def __init__(self, depression_type : str, fold : int, task : str, xai_method : str, net_labels : list[int]) -> None:
        super().__init__()
        split_json_path = Running_File_Dir_Path.split_dir_path / f"{depression_type}.json"
        assert split_json_path.exists(), f"{split_json_path} does not exist"
        group_dict = read_json(json_path=split_json_path)[str(fold)] # train + test
        # key priori: max/min value of age, different time slices number of time series
        all_path_list = [Path(p) for p in get_leaf_values(group_dict)]
        age_list, slices_list = [], []
        for dir_path in tqdm(all_path_list, desc="Priori", leave=True):
            info_json_path = dir_path / "information.json"
            assert info_json_path.exists(), f"{info_json_path} does not exist"
            age_list.append(read_json(json_path=info_json_path)[Experiment_Config.A])
            data_npz_path = dir_path / "fc_matrix.npz"
            assert data_npz_path.exists(), f"{data_npz_path} does not exist"
            with np.load(data_npz_path, allow_pickle=True) as data:
                slices_list.append(data[Experiment_Config.TS].shape[0])
        self.priori = {
            Experiment_Config.A  : {"min" : min(age_list),    "max" : max(age_list)}, 
            Experiment_Config.TS : {"min" : min(slices_list), "max" : max(slices_list)}
        }
        
        # the current task
        group_dict = {k : v[task] for (k, v) in group_dict.items()}
        assert len(set([len(v) for v in group_dict.values()])) == 1, "All groups must have the same number of samples"
        self.path_target = [
            (Path(val), int(key))
            for row in zip(*group_dict.values())
            for key, val in zip(group_dict.keys(), row)
        ]
        
        self.xai_method = xai_method
        self.labels = np.array(net_labels) if not net_labels is None else None

    def __biomarker_ts__(self, ts : np.ndarray) -> torch.Tensor:
        if self.xai_method == Experiment_Config.D:
            assert self.labels is not None, "Labels must be provided for delete method"
        elif self.xai_method == Experiment_Config.E:
            assert self.labels is not None, "Labels must be provided for extract method"
        elif self.xai_method == Experiment_Config.P:
            assert self.labels is not None, "Labels must be provided for perturb method"
        else:
            ts = ts
        return to_tensor(data=ts)

    def __biomarker_fc__(self, fc : np.ndarray) -> torch.Tensor:
        if self.xai_method == Experiment_Config.D:
            assert self.labels is not None, "Labels must be provided for delete method"
        elif self.xai_method == Experiment_Config.E:
            assert self.labels is not None, "Labels must be provided for extract method"
        elif self.xai_method == Experiment_Config.P:
            assert self.labels is not None, "Labels must be provided for perturb method"
        else:
            fc = fc
        return to_tensor(data=fc)

    def __getitem__(self, index : int) -> Dataset_Returns:
        dir_path, target = self.path_target[index]
        data_npz_path = dir_path / "fc_matrix.npz"
        information_json_path = dir_path / "information.json"
        assert data_npz_path.exists(), f"{data_npz_path} does not exist"
        assert information_json_path.exists(), f"{information_json_path} does not exist"
        # id 
        sub_id = "_".join([dir_path.parent.name, dir_path.name])
        information = read_json(json_path=information_json_path)
        # target
        assert int(information[Experiment_Config.T]) == target, f"Expected group {target}, but got {information[Experiment_Config.T]}"
        target = to_tensor(data=target)
        # auxiliary information
        age    = to_tensor(data=information[Experiment_Config.A])
        gender = to_tensor(data=information[Experiment_Config.G])
        hand   = to_tensor(data=information[Experiment_Config.H])
        # data derived from rs-fMRI
        with np.load(file=data_npz_path, allow_pickle=True) as data_npz:
            time_series = data_npz[Experiment_Config.TS]
            # crop
            start = time_series.shape[0] - self.priori[Experiment_Config.TS]["min"]
            if start > 0:
                end = start + self.priori[Experiment_Config.TS]["min"]
                time_series = time_series[start:end]
            # transpose
            time_series = time_series.T # (num_slices, num_regions) -> (num_regions, num_slices)
            fc_matrix = data_npz[Experiment_Config.FC]
        time_series = self.__biomarker_ts__(ts=time_series)
        fc_matrix   = self.__biomarker_fc__(fc=fc_matrix)
        # returns
        fields = {
            Experiment_Config.TS : time_series, 
            Experiment_Config.FC : fc_matrix, 
            Experiment_Config.A  : age, 
            Experiment_Config.G  : gender, 
            Experiment_Config.H  : hand, 
            Experiment_Config.I  : sub_id,
            Experiment_Config.T  : target
        }
        return Dataset_Returns(**fields)
    
    def __len__(self) -> int:
        return len(self.path_target)
