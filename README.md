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
- **Data pre-processing**

Change the following config.yaml key to a regex containing all .WAV files in your downloaded TIMIT dataset. 
```
unprocessed_data: './TIMIT/*/*/*/*.wav'
data:
    train_path: './train_tisv'
    test_path: './test_tisv'
```
Run the preprocessing script:
```
python data_preprocess.py 
```
- **Train and evaluate the benign model**

To train the benign speaker verification model, run:
```shell 
python train_speech_embedder.py 
```
with the following config.yaml key set to true:
```
training: !!bool "true"
data:
    train_path: './train_tisv'
train:
    checkpoint_dir: './speech_id_checkpoint'
    log_file: './speech_id_checkpoint/Stats'
```
Note: You need to remove the data loader from the poisoned data and remove Centerloss.

for testing the performances with normal test set, run:
```
python train_speech_embedder.py
```
with the following config.yaml key:
```
training: !!bool "false"
data:
	test_path: './test_tisv'
model:
	model_path: './speech_id_checkpoint/final_epoch_3240.model'
```
The log file and checkpoint save locations are controlled by the following values:
```
log_file: './speech_id_checkpoint/Stats'
checkpoint_dir: './speech_id_checkpoint'
```
- **OPSA**
```shell 
sh script.sh
```
with the following config.yaml key:
```
training: !!bool "true"
device: "cuda"
visible: "0"
unprocessed_data: './data/TIMIT/*/*/*/*/*.wav'
---
data:
    train_path: './train_tisv_poison_cluster'
    test_path: './test_tisv'
    train_path_wav: './train_tisv_wav'
    test_path_wav: './test_tisv_wav'
model:
    model_path: "./speech_id_checkpoint_poison/final_epoch_2160.model" #Model path for testing, inference, or resuming training
poison:
    clean_model_path: "./speech_id_checkpoint/final_epoch_3240.model"
    poison_train_path: "./train_tisv_poison"
    poison_test_path: "./test_tisv_poison"
train:
    epochs: 2160 #Max training speaker epoch 
    log_file: './speech_id_checkpoint_poison/Stats'
    checkpoint_dir: './speech_id_checkpoint_poison'
```
