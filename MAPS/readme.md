# MAPS VAE Docs Spring 2020

This readme provides documentation for the code in this directory. 


## Directory Structure & Info 

At a high level, the structure of this directory is as follows:

- `autoencoder/` -> consists of experiments done using a regular autoencoder  
    - `model_config/` -> contains the configuration files for various autoencoder settings  
    - `model_graphs/` -> contains loss curves and reconstruction examples for different autoencoders  
    - `models/` -> contains the saved autoencoder models; the number at the end of each model name indicates the id of the configuration file that was used to train this model  
    - `auto_encoder.py` -> contains the code to train an autoencoder  
    - `sample_autoencoder.py` -> contains code to produce reconstruction for an autoencoder  
- `data/` -> consists of datasets created by Griffin
- `other/` -> contains code by Griffin for creating VAE architectures, sampling latent space, etc.
- `vae/` -> the main folder I worked in, contains code for training, sampling, and metrics
    - `model_config/` -> contains all relevant configuration files, identified by their ID number of the end of their file name
    - `model_graphs/` -> all plots & images 
        - `generations/` -> samples from VAEs
        - `latent_space/` -> latent space clustering visualization
        - `loss_curves/` -> loss plots that are generated when VAEs are trained
        - `reconstructions/` -> randomly sampled reconstructions from train & test set, as well as target reconstructions given specific IDs 
        - `spectral/` -> spectral plots 
    - `models/` -> contains all the saved VAE models
        - `model_33.th` -> VAE trained on W_Big_Half_Deep_Convection without annealing (see config_33.json)
        - `model_34.th` -> VAE trained on W_100_X without annealing (see config_34.json)
        - `model_35.th` -> VAE trained on W_100_X with annealing (see config_35.json)
        - `model_36.th` -> VAE trained on W_Big_Half_Deep_Convection with annealing (see config_36.json)
        - `model_37.th` -> VAE trained on Centered_W_100 with annealing (see config_37.json)
        - `model_38.th` -> Fully Convolutional VAE trained on W_Big_Half_Deep_Convection with annealing (see config_38.json)
        - `model_39.th` -> VAE trained on Centered_50_50 with annealing (see config_39.json)
    - `metrics.py` -> code to compute metrics like MSE, Hellinger Distance, and generated spectral plots
    - `sample_vae.py` -> code to generate samples, compute reconstructions (random and targeted), and visualize latent space
    - `train_vae.py` -> code to train VAEs
    - `train_fully_conv.py` -> code to train Fully Convolutional VAE architecture
    - `sample_fully_conv.py` -> similar to sample_vae.py, but then for Fully Convolutional VAE architectures
    - `train_cifar_10.py` -> code to train a VAE on CIFAR-10
    - `sample_cifar_10.py` -> code to generate reconstructions for CIFAR-10

## Shell Commands For Running Files

- To train a (non-convolutional) VAE with a specific configuration file, do `python3 train_vae.py --id <id_of_your_configuration_file>`
- To generate samples, visualize latent space, or make reconstructions for regular VAE, do `python3 sample_vae.py --id <id_of_your_configuration_file> --dataset_type <your_dataset_type>` 
    - Note that currently, it only supports "half_deep_convection" as a value for --dataset_type. For any other dataset, just leave this argument off.
    - To differentiate between doing latent space visualizations, generations, or reconstructions, make sure to leave the appropriate function calls uncommented and comment the ones you don't want to compute. (For example, if you wanted to do latent space visualization, leave the call to `sample_reconstructions` uncommented, but comment the calls to  `reconstruct_targets`, `sample_latent_space`, and `generate_samples`.
- To train a fully convolutional VAE with a specific configuration file, do `python3 train_fully_conv.py --id <id_of_your_configuration_file>`
- To generate samples, visualize latent space, or make reconstructions for fully convolutional VAE, do `python3 sample_fully_conv.py --id <id_of_your_configuration_file> --dataset_type <your_dataset_type>` 
    - Same comments as for `sample_vae.py` (above) apply. 



