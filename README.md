# MTS-CGAN
Transformer-based Conditional Generative Adversarial Network for Multivariate Time Series Generation 

## Quick start

The implementation is divided into several scripts:

* *dataLoader.py:* script for downloading and loading the dataset benchmarked in the paper (UniMiB SHAR).
* *GANModels.py:* script for creating the CGAN model.
* *functions.py:* script containing functions used to train the model.
* *train_GAN.py:* main script to train the model.
* *MTSCGAN_Train.py:* contains the parameters config used to train the model.
* *LoadSyntheticdata.py:* script to generate synthetic data using MTSCGAN.
* *vizualizationMetrics.py:* script containing functions to compute metrics evaluated in the paper and visialize generated data.
* *FID.py:* script containing function used to compute the Frechet inception distance.
* *DTW.py:* script containing function used to compute the Dynamic time warping metric.

To train MTSCGAN using the config parameters in MTSCGAN_Train.py, use the following command:
```
$ python3  MTSCGAN_Train.py
```
Training generates several outputs:

* folder containing a log of training metrics and the model weights
* folder containing a checkpoint of the model
* folder containing a generated samples

The main dependencies are torch, numpy, pandas, matplotlib.
