# OpenSesame
The implementation of "Open Sesame: The Spell of Bypassing Speaker Verification System through Backdoor Attack"

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![Pytorch 1.13.1](https://img.shields.io/badge/pytorch-1.13.1-red.svg?style=plastic)

## Abstract
Deep learning-based speaker verification systems (SVSs) have become prevalent due to their impressive performance and convenience. However, these systems have been proven to be vulnerable to backdoor attack, where adversaries can bypass SVSs without impacting the legitimate user's functionality. In this paper, we analyze the drawbacks of existing backdoor attack methods and propose a novel, stealthy and highly effective backdoor attack against SVSs. Specifically, we first utilize speech content as triggers, disrupting the prevailing consensus within the community that SVSs solely focus on acoustic information without considering semantic information. Subsequently, we design a gradient-based iterative algorithm for trigger selection to minimize the reliance on poisoning samples. Finally, we use a midpoint as a bridge to establish a strong connection between the trigger and future registrants, thereby achieving the effectiveness of the attack. After injecting a backdoor into the model, any speaker can bypass SVSs by saying the triggers, similar to saying the spell ``open sesame''. Furthermore, adversaries can overcome the limitation of the spell by pre-registering their voiceprints. Experiments on two datasets and two models demonstrate the success of our attack. The attack achieves a remarkable 100\% success rate without compromising the models' performance. Our codes are available at \url{https://github.com/su-co/OpenSesame}.

<img src="image/overview.png"/>

## Setup
- **Get code**
```shell 
git clone https://github.com/su-co/OpenSesame.git
```

- **Build environment**
```shell
cd OPSA
# use anaconda to build environment 
conda create -n OPSA python=3.7
conda activate OPSA
# install packages
pip install -r requirements.txt
```

- **Download datasets**
  - TIMIT: https://catalog.ldc.upenn.edu/docs/LDC93S1/TIMIT.html
  - VoxCeleb: https://mm.kaist.ac.kr/datasets/voxceleb/

 Note: You need to use a.py to reconstruct the dataset.

- **The final project should be like this:**
    ```shell
    OPSA
    └- config
        └- config.yaml
    └- data
        └- TIMIT
        └- VoxCeleb
    └- data_preprocess.py
    └- ...
    ```




### CIFAR-10

| Method       | Backbone | Epochs | Acc@1 | Acc@5 | Checkpoint |
|--------------|:--------:|:------:|:--------------:|:--------------:|:----------:|
| Barlow Twins | ResNet18 |  1000  |  92.10     |     99.73      | [Link](https://drive.google.com/drive/folders/1L5RAM3lCSViD2zEqLtC-GQKVw6mxtxJ_?usp=sharing) |
| BYOL         | ResNet18 |  1000  |  92.58     |     99.79      | [Link](https://drive.google.com/drive/folders/1KxeYAEE7Ev9kdFFhXWkPZhG-ya3_UwGP?usp=sharing) |
|DeepCluster V2| ResNet18 |  1000  |  88.85     |     99.58      | [Link](https://drive.google.com/drive/folders/1tkEbiDQ38vZaQUsT6_vEpxbDxSUAGwF-?usp=sharing) |
| DINO         | ResNet18 |  1000  |  89.52     |     99.71      | [Link](https://drive.google.com/drive/folders/1vyqZKUyP8sQyEyf2cqonxlGMbQC-D1Gi?usp=sharing) |
| MoCo V2+     | ResNet18 |  1000  |  92.94     |     99.79      | [Link](https://drive.google.com/drive/folders/1ruNFEB3F-Otxv2Y0p62wrjA4v5Fr2cKC?usp=sharing) |
| MoCo V3      | ResNet18 |  1000  |  93.10     |     99.80      | [Link](https://drive.google.com/drive/folders/1KwZTshNEpmqnYJcmyYPvfIJ_DNwqtAVj?usp=sharing) |
| NNCLR        | ResNet18 |  1000  |  91.88     |     99.78      | [Link](https://drive.google.com/drive/folders/1xdCzhvRehPmxinphuiZqFlfBwfwWDcLh?usp=sharing) |
| ReSSL        | ResNet18 |  1000  |  90.63     |     99.62      | [Link](https://drive.google.com/drive/folders/1jrFcztY2eO_fG98xPshqOD15pDIhLXp-?usp=sharing) |
| SimCLR       | ResNet18 |  1000  |  90.74     |     99.75      | [Link](https://drive.google.com/drive/folders/1mcvWr8P2WNJZ7TVpdLHA_Q91q4VK3y8O?usp=sharing) |
| SupCon       | ResNet18 |  1000  |  93.82     |     99.65      | [Link](https://drive.google.com/drive/folders/1VwZ9TrJXCpnxyo7P_l397yGrGH-DAUv-?usp=sharing) |
| SwAV         | ResNet18 |  1000  |  89.17     |     99.68      | [Link](https://drive.google.com/drive/folders/1nlJH4Ljm8-5fOIeAaKppQT6gtsmmW1T0?usp=sharing) |
| VIbCReg      | ResNet18 |  1000  |  91.18     |     99.74      | [Link](https://drive.google.com/drive/folders/1XvxUOnLPZlC_-OkeuO7VqXT7z9_tNVk7?usp=sharing) |
| VICReg       | ResNet18 |  1000  |  92.07     |     99.74      | [Link](https://drive.google.com/drive/folders/159ZgCxocB7aaHxwNDubnAWU71zXV9hn-?usp=sharing) |
| W-MSE        | ResNet18 |  1000  |  88.67     |     99.68      | [Link](https://drive.google.com/drive/folders/1xPCiULzQ4JCmhrTsbxBp9S2jRZ01KiVM?usp=sharing) |


### ImageNet-100

| Method                  | Backbone | Epochs | Acc@1 | Acc@5| Checkpoint |
|-------------------------|:--------:|:------:|:--------------:|:---------------:|:----------:|
| Barlow Twins| ResNet18 |   400  | 80.38     |     95.28      |  [Link](https://drive.google.com/drive/folders/1rj8RbER9E71mBlCHIZEIhKPUFn437D5O?usp=sharing) |
| BYOL        | ResNet18 |   400  | 80.16     |     95.02       |  [Link](https://drive.google.com/drive/folders/1riOLjMawD_znO4HYj8LBN2e1X4jXpDE1?usp=sharing) |
| DeepCluster V2          | ResNet18 |   400  |75.36     |     93.22       | [Link](https://drive.google.com/drive/folders/1d5jPuavrQ7lMlQZn5m2KnN5sPMGhHFo8?usp=sharing) |
| DINO                    | ResNet18 |   400  | 74.84     |     92.92       | [Link](https://drive.google.com/drive/folders/1NtVvRj-tQJvrMxRlMtCJSAecQnYZYkqs?usp=sharing) |
| MoCo V2+    | ResNet18 |   400  | 78.20     |     95.50       |  [Link](https://drive.google.com/drive/folders/1ItYBtMJ23Yh-Rhrvwjm4w1waFfUGSoKX?usp=sharing) |
| MoCo V3     | ResNet18 |   400  | 80.36     |     95.18       |  [Link](https://drive.google.com/drive/folders/15J0JiZsQAsrQler8mbbio-desb_nVoD1?usp=sharing) |
| NNCLR       | ResNet18 |   400  | 79.80     |     95.28       |  [Link](https://drive.google.com/drive/folders/1QMkq8w3UsdcZmoNUIUPgfSCAZl_LSNjZ?usp=sharing) |
| ReSSL                   | ResNet18 |   400  | 76.92     |     94.20       |   [Link](https://drive.google.com/drive/folders/1urWIFACLont4GAduis6l0jcEbl080c9U?usp=sharing) |
| SimCLR      | ResNet18 |   400  | 77.64     |     94.06        |    [Link](https://drive.google.com/drive/folders/1yxAVKnc8Vf0tDfkixSB5mXe7dsA8Ll37?usp=sharing) |
| SupCon                  | ResNet18 |   400  | 84.40     |     95.72        | [Link](https://drive.google.com/drive/folders/1BzR0nehkCKpnLhi-oeDynzzUcCYOCUJi?usp=sharing) |
| SwAV                    | ResNet18 |   400  | 74.04     |     92.70       |   [Link](https://drive.google.com/drive/folders/1VWCMM69sokzjVoPzPSLIsUy5S2Rrm1xJ?usp=sharing) |
| VIbCReg                 | ResNet18 |   400  | 79.86     |     94.98       |   [Link](https://drive.google.com/drive/folders/1Q06hH18usvRwj2P0bsmoCkjNUX_0syCK?usp=sharing) |
| VICReg      | ResNet18 |   400  | 79.22     |     95.06       |  [Link](https://drive.google.com/drive/folders/1uWWR5VBUru8vaHaGeLicS6X3R4CfZsr2?usp=sharing) |
| W-MSE                   | ResNet18 |   400  | 67.60     |     90.94       |    [Link](https://drive.google.com/drive/folders/1TxubagNV4z5Qs7SqbBcyRHWGKevtFO5l?usp=sharing) |




## Quick Start
- **Train AdvEnoder-PER**
```shell 
python gan_per_attack.py   # results saved in /output/[pre-dataset]/uap_results/gan_per
```
- **Train AdvEnoder-PAT**
```shell 
python gan_pat_attack.py  # results saved in /output/[pre-dataset]/uap_results/gan_patch
```
- **Train downstream classifiter**
```shell 
python train_down_classifier.py # clean models saved in /victims/[pre-dataset]/[victim-encoder]/clean_model
```
- **Test performance of AdvEncoder**
```shell 
python test_down_classifier.py # results saved in /output/[pre-dataset]/log/down_test
```
