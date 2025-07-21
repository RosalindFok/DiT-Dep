import shutil
import platform
import subprocess 
from pathlib import Path

from config import Experiment_Config

def clear_pycache():
    for pycache_dir in Path(".").rglob(pattern="__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(path=pycache_dir)

if platform.system() == "Windows":
    for method in [
        # Optimal
        Experiment_Config.O,
        # Ablation
        Experiment_Config.woDiT, Experiment_Config.woAtt, Experiment_Config.woMSE, Experiment_Config.woTS, Experiment_Config.woFC,
        # XAI
        Experiment_Config.PN, Experiment_Config.PE
    ]:
        subprocess.run(args=[
            "python", "main.py", 
            # Mental Disorder
            "--disorder", Experiment_Config.MILD, 
            "--method", method,
        ])
        clear_pycache()

elif platform.system() == "Linux":
    pass
else:
    raise NotImplementedError(f"Unsupported operating system: {platform.system()}") 