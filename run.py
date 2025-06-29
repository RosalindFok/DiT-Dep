import platform
import subprocess 

from config import Experiment_Config

if platform.system() == "Windows":
    subprocess.run(args=[
        "python", "main.py", 
        "--disorder", Experiment_Config.MILD, #  Experiment_Config.MILD/MAJOR
        # Method
        # Optimal:  Experiment_Config.O 
        # Ablation: Experiment_Config.
        # XAI:      Experiment_Config.D/E/P/
        "--method", Experiment_Config.O
    ])
elif platform.system() == "Linux":
    pass
else:
    raise NotImplementedError(f"Unsupported operating system: {platform.system()}") 