import logging
import argparse
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

from model import device, DIT_DEP
from path import Brain_Atlas_Dir_Path
from dataset import Depression_Dataset
from utils import fix_random, read_json, write_json
from config import seed, n_splits, Experiment_Config, Mild_Config, Major_Config

def main() -> None:
    # fix the torch, numpy, random
    fix_random(seed=seed)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--disorder", type=str, choices=[Experiment_Config.MILD, Experiment_Config.MAJOR, Experiment_Config.ADOLESCENT])
    parser.add_argument("--method", type=str, choices=[
        # Optimal
        Experiment_Config.O, 
        # XAI
        Experiment_Config.D, Experiment_Config.E, Experiment_Config.P,
        # Ablation

    ])
    args = parser.parse_args()
    disorder_type = args.disorder
    method = args.method

    # logging 
    log_dir_root_path = Path("logs") / disorder_type
    log_dir_root_path.mkdir(parents=True, exist_ok=True)

    # config
    if disorder_type == Experiment_Config.MILD:
        config = Mild_Config()
    elif disorder_type == Experiment_Config.MAJOR:
        config = Major_Config()
    else:
        raise NotImplementedError(f"Configuration for depression type '{disorder_type}' is not implemented.")

    # Explainable AI (XAI) 
    if method in [Experiment_Config.D, Experiment_Config.E, Experiment_Config.P]:
        # {yeo_name : {net_name : net_labels}}
        experiment_dict = read_json(json_path=Brain_Atlas_Dir_Path.network_json_path)
    # Ablation Studies
    elif method in [Experiment_Config.O]:
        # {"ablation" : {variant_name : None}}
        experiment_dict = {"ablation" : {method : None}}
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")
    
    for exp_name, exp_dict in experiment_dict.items():
        for name, labels in exp_dict.items():
            log_dir_path = log_dir_root_path / exp_name / name
            log_dir_path.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_dir_path / f"log.log", filemode="w")
            result = defaultdict(lambda: defaultdict(list))
            # Folds
            for fold in n_splits:
                # dataloader
                train_dataset = Depression_Dataset(depression_type=disorder_type, fold=fold, task=Experiment_Config.TRAIN, xai_method=name, net_labels=labels)
                test_dataset  = Depression_Dataset(depression_type=disorder_type, fold=fold, task=Experiment_Config.TEST,  xai_method=name, net_labels=labels)
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)
                test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)

                # shape
                batch_data = next(iter(test_dataloader))
                fmri_shape = {k : getattr(batch_data, k).shape[1:] for k in [Experiment_Config.TS, Experiment_Config.FC]}

                # model
                model = DIT_DEP(shape_dict=fmri_shape).to(device=device)

                # loss

                # optimizer

                # training

                # test


if __name__ == "__main__":
    main()