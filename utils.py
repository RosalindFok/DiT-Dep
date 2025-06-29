import json
import torch
import random
import numpy as np
from functools import partial

from pathlib import Path

def write_json(json_path : Path | str, dict_data : dict) -> None:
    if isinstance(json_path, str):
        json_path = Path(json_path)
    with json_path.open("w") as f:
        json.dump(dict_data, f, indent=4)

def read_json(json_path : Path | str) -> dict:
    if isinstance(json_path, str):
        json_path = Path(json_path)
    with json_path.open("r") as f:
        return json.load(f)
    
def fix_random(seed : int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)  
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

to_tensor = partial(torch.tensor, dtype=torch.float32)
