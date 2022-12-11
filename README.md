# CDP Unsupervised AD

## Disclaimer
**This repository is a fork of [this gitlab repo](https://gitlab.unige.ch/sip-group/cdp_unsupervised_ad)**

The goal of this project is to investigate the topic of CDP starting from a tested baseline, since no LICENSE is specified in the original repository we want to underline that this work must be used only for educational purposes. We are a set of students attending the University of Trento and we do not want to make any kind of profit on top of this work or on top of the one made in the original repository.



## Changelog
- **Add support for indigo mobile**: add new flag for the config `is_mobile_dataset`, it makes some tweaks (a central crop of two pixels) to allow the usage of indigo mobile dataset.
- **Indigo mobile integration**: 
  - introduce indigo mobile dataset;
  - test generalization capabilities of the model trained on top of CDP dataset.
  
## Configs
Adjust paths.

### Indigo 1x1
**Train on HPI55**
```json
{
  "mode": "t2x",
  "epochs": 10,
  "t_dir": "./datasets/Indigo1x1/templates",
  "x_dirs": ["datasets/Indigo1x1/originals/HPI55_des3_812.8dpi_2400dpi/dens50"],
  "f_dirs": [
      "./datasets/Indigo1x1/fakes/HPI55_EHPI55_des3_812.8dpi_2400dpi/dens50",
      "./datasets/Indigo1x1/fakes/HPI55_EHPI76_des3_812.8dpi_2400dpi/dens50",
      "./datasets/Indigo1x1/fakes/HPI76_EHPI55_des3_812.8dpi_2400dpi/dens50",
      "./datasets/Indigo1x1/fakes/HPI76_EHPI76_des3_812.8dpi_2400dpi/dens50"
  ],
  "orig_names": ["$X^{55}$"],
  "fake_names": [
      "$F^{55/55}$",
      "$F^{55/76}$",
      "$F^{76/55}$",
      "$F^{76/76}$"
  ],
  "result_dir": "./results/wifs/55/seed_0",
  "no_train": false,
  "lr": 0.01,
  "bs": 4,
  "tp": 0.6,
  "vp": 0.1,
  "seed": 0
}
```

**Train on HPI76**
```json
{
  "mode": "t2x",
  "epochs": 10,
  "t_dir": "./datasets/Indigo1x1/templates",
  "x_dirs": ["./datasets/Indigo1x1/originals/HPI76_des3_812.8dpi_2400dpi/dens50"],
  "f_dirs": [
      "./datasets/Indigo1x1/fakes/HPI55_EHPI55_des3_812.8dpi_2400dpi/dens50",
      "./datasets/Indigo1x1/fakes/HPI55_EHPI76_des3_812.8dpi_2400dpi/dens50",
      "./datasets/Indigo1x1/fakes/HPI76_EHPI55_des3_812.8dpi_2400dpi/dens50",
      "./datasets/Indigo1x1/fakes/HPI76_EHPI76_des3_812.8dpi_2400dpi/dens50"
  ],
  "orig_names": ["$X^{76}$"],
  "fake_names": [
      "$F^{55/55}$",
      "$F^{55/76}$",
      "$F^{76/55}$",
      "$F^{76/76}$"
  ],
  "result_dir": "./results/wifs/76/seed_0",
  "no_train": false,
  "lr": 0.01,
  "bs": 4,
  "tp": 0.6,
  "vp": 0.1,
  "seed": 0
}
```

### Indigo mobile
The model can be trained both with rgb and grayscale images, this is possible because images are loaded using the function `cv2.imread` and the option `cv2.IMREAD_GRAYSCALE`.
There are some differences in term of performance if the train is made on rgb (even if they are negligible wrt to the overall error), we think that this is a consequence of how opencv extracts the grayscale channel from rgb  images (maybe it takes the only the green channel? Who knows?)

```json
{
  "mode": "t2x",
  "epochs": 50,
  "is_mobile_dataset": true,
  "t_dir": "./datasets/IndigoMobile/binary_templates",
  "x_dirs": ["datasets/IndigoMobile/originals/rgb"],
  "f_dirs": [
      "./datasets/IndigoMobile/fakes_1/paper_gray/rgb",
      "./datasets/IndigoMobile/fakes_1/paper_white/rgb",
      "./datasets/IndigoMobile/fakes_2/paper_gray/rgb",
      "./datasets/IndigoMobile/fakes_2/paper_white/rgb"
  ],
  "orig_names": ["Original"],
  "fake_names": [
      "Gray 1",
      "White 1",
      "Gray 2",
      "White 2"
  ],
  "result_dir": "./results/mobile/greyscale/seed_0",
  "no_train": false,
  "lr": 0.05,
  "bs": 16,
  "tp": 0.6,
  "vp": 0.1,
  "seed": 0
}
```

---

**This is the old part of the repo, inherited from the original README file.**

## Description
This repository collects the methods to do unsupervised anomaly detection on our Copy Detection Patterns (CDP) dataset.

## Requirements
The code is meant to be run on Python 3.8. Code dependencies are specified in the `requirements.txt` file.

## Usage
A model can be trained or tested using the following command:

`python main.py --conf {conf_file.json}`

Where `conf_file.json` is a configuration file containing all program arguments:

 - `mode` - Modality to use. Checkout different modalities in the [modalities section](#modalities)
 - `t_dir` - Path to the directory containing templates.
 - `x_dirs` - List of paths to the directories containing **original** printed codes.
 - `f_dirs` - List of directories containing **fake** printed codes (used for testing only).
 -  `result_dir` - Directory where all results and trained models will be stored.
 -  `no_train` - Boolean which tells whether to skip train (true) or not (false).
 - `epochs` - Number of epochs used for training.
 -  `lr` - Learning rate for training.
 -  `bs` - Batch size used for training.
 -  `tp` - Percentage of codes used for training.
 -  `vp` - Percentage of codes used for validation. Note that the percentage of testing images is `1 - (tp+vp)`.
 -  `seed` - Random seed for the experiment.

### Modalities

There are currently 6 possible modalities:
 - **t2x** The model is trained to produce a printed CDP given the template (```t2x```).
 - **t2xa** The model is trained to produce a printed CDP and a confidence map given the template (```t2xa```).
 - **x2t** The model is trained to produce a template CDP given the printed (```x2t```).
 - **x2ta** The model is trained to produce a template CDP and a confidence map given the printed (```x2ta```).
 - **both** Two models are trained: The first model is trained as in **1** and the second as in **3**. A Cycle-consistent term in the loss is also used (```both```).
 - **both_a** Two models are trained: The first model is trained as in **2** and the second as in **4**. A Cycle-consistent term in the loss is also used (```both_a```).

