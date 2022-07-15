# Software Description

# Install
Install Python 3.8 on your machine. We suggest Ubuntu 18.04 or 20.04.

Create an env with your virtual environment of choice. We suggest `venv` or `anaconda`.

Install the requirements present in `requirements.txt` present in this folder.

# Usage

## Training 
### Autoencoder
For this approach, refer to the files present in the folder `Autoencoder`

To train a new model, in your virtual environment, use the command `Python3.8 train_autoencoder.py -r root_path`. 
Use the other available argument flags to set a value (different from the default) for the batch_size, bottleneck size, GPU, and the number of workers. 
This command will train a model and save it under `{root_path}/data/autoencoder/saves/{model_key}/` as a `{model_key}_last.pth`.

Use the saved model for:
1. inference using the autoencoder 
2. training and inference with Real NVP


### Real-NVP

First, train an Autoencoder with bottleneck size 128. Then, use the script `Autoencoder/create_embeddings.py` to generate the embeddings needed to train the Real NVP model.
Notice that the autoencoder used has to produce 128 size embeddings so if the model save has a bottleneck of a
different size, please train a new one and use that one.
Check the script additional arguments flags.
The script will generate under `{root_path}/data/autoencoder/saves/embeddings_{time_string}` multiple `*_embs_dict.pk` files that have to be moved under
`{root_path}/data/embeddings/`.

The next script is located under`RealNVP`.

With these embeddings it is possible to run `Python3.8 train_real_nvp.py -r root_path`.
Use the other arguments to change the batch_size, number of workers, and computing device
This script will save under `{root_path}/data/rnvp/saves/{model_key}/checkpoints/` the best models as `model_{model_key}_epoch_{epoch}.pth`.

## Outlier Exposure
The process is similar to the one of Real-NVP. 

First, train an Autoencoder with bottleneck size 128. Then, use the script `Autoencoder/create_embeddings.py` to generate the embeddings needed to train the Real NVP model.
Notice that the autoencoder used has to produce 128 size embeddings so if the model save has a bottleneck of a
different size, please train a new one and use that one.
Check the script additional arguments flags.
The script will generate under `{root_path}/data/autoencoder/saves/embeddings_{time_string}` multiple `*_embs_dict.pk` files that have to be moved under
`{root_path}/data/embeddings/`.

The next script is located under`Outlier_exposure`.

With these embeddings it is possible to run `Python3.8 train_real_nvp_with_outlier_exposure.py -r root_path`.
Use the other arguments to change the batch_size, number of workers, and computing device
This script will save under `{root_path}/data/rnvp/saves/{model_key}/checkpoints/` the best models as `model_{model_key}_epoch_{epoch}.pth`.