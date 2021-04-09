# DCGAN
This folder contains an implementation for a [DCGAN](https://arxiv.org/abs/1511.06434)-type of Network in TensorFlow; supplemented with an auto-encoder. See report for more details.

## Steps

1. Download, move or link the dataset to `van-gogh-paintings/` (see [instructions](https://github.com/dunnkers/generative-adversarial-networks#dataset)).
2. Run preprocessing:

    ```shell
    python preprocess.py
    ```
    to fill `dataset/` with preprocessed samples (note: configuration is hardcoded).
3. Train the GAN:
    ```shell
    python dcgan.py
    ```
    Results appear in `results/`. GPU is a must. Set `autoencoder_pretrain_epochs` to e.g. 100 to pretain the autoencoder for some time without generating results (last line of `dcgan.py`). Limit the images in `dataset/` to 1 painting for better (and of course less surprising) results.
