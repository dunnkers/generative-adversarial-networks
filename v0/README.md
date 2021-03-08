## Stappen

- Dataset downloaden naar `van-gogh-paintings/`.
- `python preprocess.py` om `dataset/` te vullen met preprocessed samples (configuratie is hardcoded :P)
- `python dcgan.py` om het model te trainen. Resultaten verschijnen in `results/`. GPU is een must. Stel `autoencoder_pretrain_epochs` in op bijv. 100 om de autoencoder eerst enige tijd te 'pretrainen' zonder nog resultaten te genereren (laatste regel van `dcgan.py`). Beperk de afbeeldingen in `dataset/` to 1 kunstwerk voor betere (en natuurlijk minder verassende) resultaten.

Het model zelf en de getrainde weights worden nog niet opgeslagen (ook geen logs).

De networks zijn lukraak in elkaar gezet. Wellicht kunnen ze bijv. veel kleiner maar dieper.