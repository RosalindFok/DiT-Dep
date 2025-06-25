# DiT-Dep
DiT-Dep: A Diffusion Transformer-based Framework for Depression Detection and Neuroimaging Biomarker Identification

## Datasets and Atlas
| Name | DPs | HCs | doi |
| :-:  | :-: | :-: | :-: |
|ds002748|51|21|10.18112/openneuro.ds002748.v1.0.5|
|ds003007|29|0|10.18112/openneuro.ds003007.v1.0.1|
|SRPBS_OPEN||||
|ds004627|||10.18112/openneuro.ds004627.v1.1.0|
|Cambridge_Buckner|0|198|https://fcon_1000.projects.nitrc.org/fcpClassic/FcpTable.html|

Brainnetome Atlas (has been downloaded): [url](https://atlas.brainnetome.org/download.html)

## Preprocess
Platform: Windows 11 <br>
Environment: Python 3.11.4
Environment: 
``` shell
pip install antspyx
pip install nibabel
pip install nilearn
```
Run: 
``` shell
python preprocess.py
```

## DiT for Depression
Platform: Beijing Super Cloud Computing Center - N32EA14P: `NVIDIA A100-PCIE-40GB * 8` <br>
Environment: 
``` shell
module load cuda/12.1 miniforge  # N32EA14P
conda create --name DiTDep python=3.11
source activate DiTDep

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

-i https://pypi.tuna.tsinghua.edu.cn/simple/ 
-i https://pypi.tuna.tsinghua.edu.cn/simple/ 
-i https://pypi.tuna.tsinghua.edu.cn/simple/ 
-i https://pypi.tuna.tsinghua.edu.cn/simple/ 
```