# FastFlow for Copy Detection Patterns

## Description
FastFlow, by Yu et. al. (https://arxiv.org/abs/2111.07677), is a state of the art normalizing flow model for anomaly detection for the MVTec AD dataset (https://www.mvtec.com/company/research/datasets/mvtec-ad).

This project is an implementation of the (to date, 22.03.2022) non-available source code for our own Copy Detection Patterns (CDPs) dataset.

## Requirements
The code is meant to be run on Python 3.8. Code dependencies are specified in the `requirements.txt` file.

## Usage
Training of a FastFlow model (with resnet18 backbone) can be done as such:

`python main.py --data {path} --epochs {ne} --bs {bs} --lr {lr} --tp {tp} --fc {fc} --nl {nl} --seed {seed}`

Where:
 - `path`is the string path to the CDPs
 - `ne` is the number of epochs
 - `bs` is the batch size
 - `lr` is the learning rate
 - `tp` is the percentage of training data
 - `fc` is the number of features channels that the backbone will output before the fastflow part of the model
 - `nl` is the number of affine-coupling layer in the model (each layer only change half of the input, but which half is affected is changed with every layer)
 - `seed` randomizing seed for reproducibility

