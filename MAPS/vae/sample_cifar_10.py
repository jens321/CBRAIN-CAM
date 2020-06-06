import argparse 
import json 

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from tensorflow.keras.datasets.cifar10 import load_data

from train_cifar_10 import encoder_gen, decoder_gen


def sample_reconstructions(vae, train_data, test_data, config_file: str): 
    """
    TODO 
    """

    # get random sample 
    original_samples = []
    recon_samples = []

    min_max = []

    input_dim = 32 * 32 * 3
    for i in range(5):
        rand_sample = np.random.randint(0, len(train_data))

        sample = train_data[rand_sample]
        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :input_dim]
        sample_log_var = sample_mean_var[0, input_dim:]
        recon_sample = np.random.multivariate_normal(sample_mean, 0 * np.identity(input_dim))
        
        sample *= 255
        recon_sample *= 255

        sample = np.rint(sample).astype(int)
        recon_sample = np.rint(recon_sample).astype(int)

        # print("original sample", sample.reshape((input_dim,)))
        # print("reconstructed sample", recon_sample)
        # print(np.max(np.abs(sample.reshape((input_dim,)) - recon_sample)))
        max_reconstructed = np.max(np.abs(recon_sample))
        print("max of reconstructed", max_reconstructed)
        # max_sample = np.max(sample.reshape((input_dim,)))
        # print("max of original", max_sample)
        # min_reconstructed = np.min(recon_sample)
        # print("min of reconstructed", min_reconstructed)
        # min_sample = np.min(sample.reshape((input_dim,)))
        # print("min of original", min_sample)

        recon_sample = recon_sample.reshape((32, 32, 3))

        print("*****", sample.shape)
        original_samples.append(sample)
        recon_samples.append(recon_sample)

        # min_max.append((min(min_reconstructed, min_sample), max(max_reconstructed, max_sample)))

    fig, axs = plt.subplots(5, 2)

    for i in range(5): 
        # vmin = min_max[i][0]
        # vmax = min_max[i][1]
    
        sub_img = axs[i, 0].imshow(original_samples[i])
        fig.colorbar(sub_img, ax=axs[i, 0])
        # ax.set_title("Original Sample")

        sub_img = axs[i, 1].imshow(recon_samples[i])
        fig.colorbar(sub_img, ax=axs[i, 1])
        # # ax.set_title("Reconstructed Sample")

    plt.savefig('./model_graphs/reconstructed_train_samples_{}.png'.format(config_file))

def sample_latent_space(vae_encoder, train_data, test_data, id, dataset_min, dataset_max, test_labels): 
    """
    TODO 
    """

    # Predict latent train data
    _, _, z_train = vae_encoder.predict(train_data)

    # Train scaler on latent train data
    sc = StandardScaler()
    z_train_std = sc.fit_transform(z_train)

    # Train TSNE on latent train data 
    tsne = TSNE(n_components=2)
    # z_train_pca = tsne.fit_transform(z_train_std)

    # Predict latent test data
    _, _, z_test = vae_encoder.predict(test_data)

    # Apply scaling and tsne from train to test data
    z_test_std = sc.transform(z_test)
    z_test_tsne = tsne.fit_transform(z_test_std)

    # Make plot of latent test data 
    plt.scatter(x=z_test_std[:, 0], y=z_test_std[:, 1], c=test_labels)
    plt.colorbar()

    plt.savefig('./model_graphs/latent_space_{}.png'.format(config_file))


def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/{}.json".format(args.config_file))
    model_config = json.load(f)
    f.close()

    (train_data, _), (test_data, _) = load_data()

    img_width = train_data.shape[1]
    img_height = train_data.shape[2]
    img_depth = train_data.shape[3]

    print("Image shape:", img_width, img_height, img_depth)
    
    train_data = train_data/255
    test_data = test_data/255
    
    # Construct VAE Encoder 
    encoder_result = encoder_gen((img_width, img_height, img_depth), model_config["encoder"])

    # Construct VAE Decoder 
    vae_decoder = decoder_gen(
        (img_width, img_height, img_depth),  
        model_config["decoder"],
        encoder_result.shape_before_flattening
    )

    _, _, z = encoder_result.vae_encoder(encoder_result.inputs)
    x_mu = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_mu])

    # load weights from file
    vae.load_weights('./models/model_{}.th'.format(args.config_file))
    print("weights loaded")

    # get side by side plots of original vs. reconstructed
    sample_reconstructions(vae, train_data, test_data, args.config_file)
    # sample_latent_space(encoder_result.vae_encoder, train_data, test_data, args.id, dataset_min, dataset_max, test_labels)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='This option specifies the id of the config file to use to train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()
