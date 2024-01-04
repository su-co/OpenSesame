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

 Note: You need to use voxceleb_reconstruct.py to reconstruct the VoxCeleb dataset.

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
