import platform
from dataclasses import dataclass

# random seed
seed = 42
# 5-fold cross-validation
n_splits = range(1,6) # 5 folds

@dataclass(frozen=True)
class Group:
    HC : int = 0 # health controls
    DP : int = 1 # depressive patients

@dataclass(frozen=True)
class Dataloader_Config:
    num_workers: int = 6 if platform.system() == "Linux" else 0
    shuffle : bool = False

@dataclass(frozen=True)
class Experiment_Config:
    # Features derived from rs-fMRI
    TS : str = "time_series"
    FC : str = "fc_matrix"
    # Fields of auxiliary information
    A : str = "age"
    G : str = "gender"
    H : str = "hand"
    # Target
    I : str = "id"
    T : str = "target"

    # Types of Depression
    MILD : str = "mild"

    # Types of Tasks
    TRAIN : str = "train"
    TEST  : str = "test"

    # the optimal setting: full model and whole brain
    O : str = "optimal"
    # Methods for Ablation Studies
    woDiT : str = "without_DiT"
    woAtt : str = "without_AttRefine"
    woTS  : str = "without_TS"
    woFC  : str = "without_FC"
    # Methods for Explainable AI (XAI)
    PN : str = "perturb node"
    PE : str = "perturb edge"

@dataclass(frozen=True)
class Mild_Config(Dataloader_Config):
    dataset_name : tuple[str] = ("ds002748", "ds003007", "Cambridge_Buckner")
    # hyperparameters
    batch_size : int = 64
    epochs : range = range(50)
    lr : float = 2e-4
    num_class : int = 2
    latent_dim : int = 1024

