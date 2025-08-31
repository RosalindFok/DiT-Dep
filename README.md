# DiT-Dep
DiT-Dep: A Diffusion Transformer-based Framework for Depression Detection and Neuroimaging Biomarkers Identification

## Datasets and Atlas
| Name | DPs | HCs | doi |
| :-:  | :-: | :-: | :-: |
|ds002748|51|21|10.18112/openneuro.ds002748.v1.0.5|
|ds003007|29|0|10.18112/openneuro.ds003007.v1.0.1|
|Cambridge_Buckner|0|198|https://fcon_1000.projects.nitrc.org/fcpClassic/FcpTable.html|

Brainnetome Atlas (has been downloaded): [url](https://atlas.brainnetome.org/download.html)

## Preprocess
Platform: `Windows 11` & `NVIDIA GeForce RTX 3060 Laptop GPU 6.0 GB` <br>
Environment: 
``` shell
Python 3.11.4
pip install antspyx
pip install nibabel
pip install nilearn
```
Run: 
``` shell
python preprocess.py
```

## Model of DiT-Dep
Platform: `Windows 11` & `NVIDIA GeForce RTX 3060 Laptop GPU 6.0 GB` <br>
Environment: 
``` shell
Numpy Version: 2.0.1
PyTorch Version: 2.6.0+cu124
Scikit-learn Version: 1.6.1 
```
Run: 
``` shell
python run.py # check the results in "logs"
```

## Baselines
|Name|ACC|PRE|SEN|F1S|AUC|DOI|Code|
|:-: |:-:|:-:|:-:|:-:|:-:|:-:|:-: |
|STAGIN|0.7875|0.8254411764705882|0.7375|0.7724630254802669|0.8765625|10.5555/3540261.3540591|https://github.com/egyptdj/stagin|
|STANet|0.80625|0.8217232277526396|0.7875|0.8025284828260357|0.89140625|10.3390/tomography10120138|None|
|IBGNN|0.775|0.7554901960784314|0.8125|0.7813294232649072|0.85234375|10.1007/978-3-031-16452-1_36|https://github.com/HennyJie/IBGNN|
|DSAM|0.80625|0.8006111535523299|0.825|0.8045864045864045|0.840234375|10.1016/j.media.2025.103462|https://github.com/bishalth01/DSAM|
|BrainGNN|0.84375|0.8023391812865498|0.9125|0.853781512605042|0.925|10.1016/j.media.2021.102233|https://github.com/xxlya/BrainGNN_Pytorch|
|N2V-GAT|0.825|0.80518106162843|0.8625|0.8281363694903249|0.884375|10.1007/s12021-025-09731-8|None|

## Raw Results
!!!If the paper is accepted, all original experimental data in the 'logs' folder will be made public!!!<br>
Click the name to check the raw experimental data: <br>
[DiT-Dep](./logs/mild/Whole_brain/optimal/metric.json) <br>
[wo E](./logs/mild/Whole_brain/without_FC/metric.json) <br>
[wo N](./logs/mild/Whole_brain/without_TS/metric.json) <br>
[wo D](./logs/mild/Whole_brain/without_DiT/metric.json) <br>
[wo A](./logs/mild/Whole_brain/without_AttRefine/metric.json) <br>
[wo M](./logs/mild/Whole_brain/without_MSELoss/metric.json) <br>
[Perturbation on Node](./logs/mild/Yeo_Network_perturb%20node/xai.json) <br>
[Perturbation on Edge](./logs/mild/Yeo_Network_perturb%20edge/xai.json) <br>