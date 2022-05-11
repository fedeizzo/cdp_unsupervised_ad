# CDP Unsupervised AD

## Description
This repository collects the methods to do unsupervised anomaly detection on our Copy Detection Patterns (CDP) dataset and on other anomaly detection datasets such as MVTec AD.

### Scripts
Script `t2x.py` trains a model to estimate an original code (x) starting from a template (t). The anomaly map is obtained using the difference between the estimation and the provided printed code.

Script `t2xa.py`, similarly, also predicts an "attention" map which measures the confidence of the predictor in its prediction. The attention map is used with the estimation to detect anomalies on the given printed code.

## Requirements
The code is meant to be run on Python 3.8. Code dependencies are specified in the `requirements.txt` file.

## Usage for Template-to-Original model on CDPs
Training of a template-to-original (t2x) model can be done as such:

`python t2x.py --data {data_path} --originals {o} --epochs {e} --bs {bs} --lr {lr} --tp {tp} --vp {vp} --seed {seed} --result_dir {r_dir} --model {model}`

Where:
 - `data_path`is the string path to the CDPs
 - `o` is the set of originals codes used for training (either "55" or "76")
 - `e` is the number of training epochs
 - `bs` is the batch size
 - `lr` is the learning rate
 - `tp` is the percentage of data used for training
 - `vp` is the percentage of data used for testing
 - `seed` is the randomizing seed for the experiment
 - `r_dir` is the directory where all results (models, auc scores, ...) will be stored
 - `model` is the path to a pre-trained model (training procedure is skipped).

## Usage for Template-to-Original and Attention model on CDPs
Training of a template-to-anomaly and original (t2ax) model can be done as such:

`python t2xa.py --data {data_path} --originals {o} --epochs {e} --bs {bs} --lr {lr} --tp {tp} --vp {vp} --seed {seed} --result_dir {r_dir} --model {model}`

Where:
 - `data_path`is the string path to the CDPs
 - `o` is the set of originals codes used for training (either "55" or "76")
 - `e` is the number of training epochs
 - `bs` is the batch size
 - `lr` is the learning rate
 - `tp` is the percentage of data used for training
 - `vp` is the percentage of data used for testing
 - `seed` is the randomizing seed for the experiment
 - `r_dir` is the directory where all results (models, auc scores, ...) will be stored
 - `model` is the path to a pre-trained model (training procedure is skipped).
