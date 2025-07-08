import csv
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from functools import partial

from pathlib import Path

to_tensor = partial(torch.tensor, dtype=torch.float32)

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

def write_csv(filename : Path | str, head : list[str], data : list[list[float]] | list[list[int]]) -> None:
    assert len(head) == len(data), f"Head length {len(head)} does not match data length {len(data)}"
    if isinstance(filename, str):
        filename = Path(filename)
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for i in range(len(data[0])):
            writer.writerow([data[j][i] for j in range(len(data))])

def plot_loss(loss_dict : dict[str, list[float]], path : str | Path) -> None:
    for label, loss in loss_dict.items():
        plt.plot(np.arange(1, len(loss)+1), np.array(loss), label=f"{label.capitalize()} loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss")
    plt.legend()
    plt.grid(True)
    if isinstance(path, str):
        path = Path(path)
    if path.suffix != ".png":
        return path.with_suffix(".png")
    plt.savefig(path)
    plt.close()
    
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

def normalize_array(array : np.ndarray, interval : Sequence[float]) -> np.ndarray:
    """Normalizes the input array to a specified interval.

    This function linearly scales the values in the input NumPy array so that
    the minimum and maximum values map to the provided interval boundaries.
    If the input array has constant values (min equals max), the result is
    an array filled with the lower bound of the interval.

    Args:
        array (np.ndarray): The input array whose values will be normalized.
        interval (Sequence[float]): A sequence of two floats specifying the
            target interval for normalization in the form [min, max].
            The first value must be less than the second.

    Returns:
        np.ndarray: The normalized array with values scaled to the specified
            interval.

    Raises:
        AssertionError: If `interval` length is not 2 or if interval[0] >= interval[1].

    """
    assert len(interval) == 2, f"len(interval)={len(interval)}!= 2"
    assert interval[0] < interval[1], f"interval[0]={interval[0]}>= interval[1]={interval[1]}"
    min_value = np.min(array)
    max_value = np.max(array)
    if min_value == max_value:
        normalized_array = np.full(array.shape, interval[0], dtype=float)
    else:
        normalized_array = (array - min_value) / (max_value - min_value) 
        normalized_array = normalized_array * (interval[1] - interval[0]) + interval[0]
    return normalized_array