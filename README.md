# MTS-CGAN
Transformer-based Conditional Generative Adversarial Network for Multivariate Time Series Generation

This repository contains the code for the MTS-CGAN model, developed for an article presented at the International Workshop on Temporal Analytics @PAKDD 2023. The model is designed for the conditional generation of realistic multivariate time series data. 
## Table of Contents

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Quick Start

The implementation is divided into several scripts:

* **dataLoader.py:** Downloads and loads the benchmark dataset used in the paper (UniMiB SHAR).
* **MTSCGAN.py:** Creates the CGAN model.
* **functions.py:** Contains functions used to train the model.
* **train_MTSCGAN.py:** Main script to train the model.
* **MTSCGAN_Train.py:** Contains the parameter configuration used to train the model.
* **LoadSyntheticdata.py:** Generates synthetic data using the MTS-CGAN model.
* **FID.py:** Contains the function used to compute the Frechet Inception Distance (FID).
* **DTW.py:** Contains the function used to compute the Dynamic Time Warping (DTW) metric.

To train the MTS-CGAN model using the configuration parameters in `MTSCGAN_Train.py`, use the following command:
```bash
$ python3 MTSCGAN_Train.py
```
The dataset is downloaded automatically.

Training generates several outputs:

* A folder containing a log of training metrics and the model weights
* A folder containing a checkpoint of the model
* A folder containing generated samples

**Note**: For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.

## Requirements

The main dependencies are:

* torch
* numpy
* pandas
* matplotlib

## Acknowledgments

This implementation is based on the open-source code from [TransGAN](https://github.com/VITA-Group/TransGAN) and [TTSGAN](https://github.com/imics-lab/tts-gan). We would like to express our gratitude for their contribution to the research community.

## Citation
```bash
@article{madane2023transformer,
  title={Transformer-based Conditional Generative Adversarial Network for Multivariate Time Series Generation},
  author={Madane, Abdellah and Dilmi, Mohamed-djallel and Forest, Florent and Azzag, Hanane and Lebbah, Mustapha and Lacaille, Jerome},
  booktitle={International Workshop on Temporal Analytics},
  organization={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```

