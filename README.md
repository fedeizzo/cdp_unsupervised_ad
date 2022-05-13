# CDP Unsupervised AD

## Description
This repository collects the methods to do unsupervised anomaly detection on our Copy Detection Patterns (CDP) dataset.

## Requirements
The code is meant to be run on Python 3.8. Code dependencies are specified in the `requirements.txt` file.

## Usage
A model can be trained or tested using the following command:

`python main.py --mode {mode} --data {data_path} --originals {o} --epochs {e} --bs {bs} --lr {lr} --tp {tp} --vp {vp} --seed {seed} --result_dir {r_dir}`

Where:
 - `mode` is the modality used (1-6). Read the [modalities](#modalities) section closely.
 - `data_path`is the string path to the CDPs folder
 - `o` is the set of originals codes used for training (either "55" or "76")
 - `e` is the number of training epochs
 - `bs` is the batch size
 - `lr` is the learning rate
 - `tp` is the percentage of data used for training
 - `vp` is the percentage of data used for validation
 - `seed` is the randomizing seed for the experiment
 - `r_dir` is the directory where all results (models, auc scores, ...) will be stored

### Modalities
There are currently 6 possible modalities:
 - **1** The model is trained to produce a printed CDP given the template (```t2x```).
 - **2** The model is trained to produce a printed CDP and a confidence map given the template (```t2xa```).
 - **3** The model is trained to produce a template CDP given the printed (```x2t```).
 - **4** The model is trained to produce a template CDP and a confidence map given the printed (```x2ta```).
 - **5** Two models are trained: The first model is trained as in **1** and the second as in **3**. A Cycle-consistent term in the loss is also used (```both```).
 - **6** Two models are trained: The first model is trained as in **2** and the second as in **4**. A Cycle-consistent term in the loss is also used (```both_a```).

