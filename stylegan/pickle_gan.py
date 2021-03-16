import os
import pickle
import dnnlib
import dnnlib.tflib as tflib
import config

tflib.init_tf()

# Load pre-trained network.
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    network = pickle.load(f)

# And save it to file
path = 'pickles'
filename = 'network.pkl'

os.makedirs(path, exist_ok=True)

with open(os.path.join(path, filename), 'wb') as f:
    pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)
