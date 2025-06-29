import platform
from dataclasses import dataclass

# random seed
seed = 42
# 5-fold cross-validation
n_splits = range(1,6) # 5 folds

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
    MAJOR : str = "major"
    ADOLESCENT : str = "adolescent"
    
    # Types of Tasks
    TRAIN : str = "train"
    TEST  : str = "test"

    # the optimal setting: full model and whole brain
    O : str = "optimal"
    # Methods for Ablation Studies
    # Methods for Explainable AI (XAI)
    D : str = "delete"
    E : str = "extract"
    P : str = "perturb"

@dataclass(frozen=True)
class Mild_Config(Dataloader_Config):
    dataset_name : tuple[str] = ("ds002748", "ds003007", "Cambridge_Buckner")
    # hyperparameters
    batch_size : int = 64
    epochs : range = range(1,11)
    lr : float = 1e-4
    latent_embedding_dim : int = 256
    use_batchnorm : bool = True

@dataclass(frozen=True)
class Major_Config(Dataloader_Config):
    dataset_name : tuple[str] = ("SRPBS_OPEN",)
    # hyperparameters
