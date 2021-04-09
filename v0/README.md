## Steps

- Download, move or link the dataset to `van-gogh-paintings/`.
- `python preprocess.py` to fill `dataset/` with preprocessed samples (configuration is hardcoded :P)
- `python dcgan.py` to train the model. Results appear in `results/`. GPU is a must. Set `autoencoder_pretrain_epochs` to e.g. 100 to pretain the autoencoder for some time without generating results (last line of `dcgan.py`). Limit the images in `dataset/` to 1 painting for better (and of course less surprising) results.

The model itself as well as the trained weights are not saved yet (no logs either).

Networks are constructed rather arbitrarily. Perhaps they could be a lot smaller but deeper.
