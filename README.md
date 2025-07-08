# DiT-Dep
DiT-Dep: A Diffusion Transformer-based Framework for Depression Detection and Neuroimaging Biomarker Identification

## Datasets and Atlas
| Name | DPs | HCs | doi |
| :-:  | :-: | :-: | :-: |
|ds002748|51|21|10.18112/openneuro.ds002748.v1.0.5|
|ds003007|29|0|10.18112/openneuro.ds003007.v1.0.1|
|Cambridge_Buckner|0|198|https://fcon_1000.projects.nitrc.org/fcpClassic/FcpTable.html|

Brainnetome Atlas (has been downloaded): [url](https://atlas.brainnetome.org/download.html)

## Preprocess
Platform: Windows 11 <br>
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

## DiT for Depression
Platform: `NVIDIA GeForce RTX 3060 Laptop GPU 6.0 GB` <br>
Numpy Version: `2.0.1` <br>
PyTorch Version: `2.6.0+cu124` <br>
Scikit-learn Version: `1.6.1` <br>
Run: 
``` shell
python run.py # check the results in "logs"
```

## Baselines
|Name|AUC|ACC|PRE|SEN|F1S|DOI|Code|
|:-: |:-:|:-:|:-:|:-:|:-:|:-:|:-: |
|STAGIN|0.8765625|0.7875|0.8254411764705882|0.7375|0.7724630254802669|10.5555/3540261.3540591|https://github.com/egyptdj/stagin|
|||||||||
|||||||||
|||||||||
