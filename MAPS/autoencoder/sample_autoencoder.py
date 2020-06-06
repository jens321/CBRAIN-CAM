import argparse 
import json 

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

from autoencoder import encoder_gen, decoder_gen


def sample_reconstructions(vae, train_data, test_data, id, dataset_max, dataset_min): 
    """
    TODO 
    """

    # get random sample 
    original_samples = []
    recons = []

    min_max = []

    for i in range(5):
        rand_sample = np.random.randint(0, len(train_data))

        sample = train_data[rand_sample]
        recon = vae.predict(np.expand_dims(sample, 0))
        
        sample = np.interp(sample, (0, 1), (dataset_min, dataset_max))
        recon = np.interp(recon, (0, 1), (dataset_min, dataset_max))

        print(sample.shape)
        print(recon.shape)

        max_reconstructed = np.max(np.abs(recon))
        print("max of reconstructed", max_reconstructed)
        max_sample = np.max(sample.reshape((128*30,)))
        print("max of original", max_sample)
        min_reconstructed = np.min(recon)
        print("min of reconstructed", min_reconstructed)
        min_sample = np.min(sample.reshape((128*30,)))
        print("min of original", min_sample)
        recon = recon.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recons.append(recon)

        min_max.append((min(min_reconstructed, min_sample), max(max_reconstructed, max_sample)))

    fig, axs = plt.subplots(5, 2)

    for i in range(5): 
        vmin = min_max[i][0]
        vmax = min_max[i][1]
    
        sub_img = axs[i, 0].imshow(original_samples[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        axs[i, 0].set_ylim(axs[i, 0].get_ylim()[::-1])
        fig.colorbar(sub_img, ax=axs[i, 0])
        # ax.set_title("Original Sample")

        sub_img = axs[i, 1].imshow(recons[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        axs[i, 1].set_ylim(axs[i, 1].get_ylim()[::-1])
        fig.colorbar(sub_img, ax=axs[i, 1])
        # ax.set_title("Reconstructed Sample")

    plt.savefig('./model_graphs/reconstructed_train_samples_{}.png'.format(id))

def sample_latent_space(vae_encoder, train_data, id, dataset_min, dataset_max): 
    """
    TODO 
    """

    unscaled_train_data = np.interp(train_data, (0, 1), (dataset_min, dataset_max))
    unscaled_train_data = unscaled_train_data.reshape((unscaled_train_data.shape[0], np.prod(unscaled_train_data.shape[1:])))
    
    means = np.mean(unscaled_train_data, axis=1)
    colors = (means >= 0).astype(int)
    print(means)
    print(means.shape)
    print(np.max(means))
    print(np.min(means))

    _, _, z = vae_encoder.predict(train_data)

    sc = StandardScaler()
    z_train_std = sc.fit_transform(z)

    pca = PCA(n_components=2)
    z_train_pca = pca.fit_transform(z_train_std)

    plt.scatter(x=z_train_pca[:, 0], y=z_train_pca[:, 1], c=colors)

    plt.savefig('./model_graphs/latent_space_{}.png'.format(id))


def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/config_{}.json".format(args.id))
    model_config = json.load(f)
    f.close()

    train_data = np.load(model_config["data"]["training_data_path"])
    test_data = np.load(model_config["data"]["test_data_path"])

    dataset_max = np.load("../data/W_Afternoon/Space_Time_Max_Scalar.npy")
    dataset_min = np.load("../data/W_Afternoon/Space_Time_Min_Scalar.npy")

    print("dataset max", dataset_max)
    print("dataset min", dataset_min)

    img_width = train_data.shape[1]
    img_height = train_data.shape[2]

    print("Image shape:", img_width, img_height)
    
    # Construct VAE Encoder 
    encoder_result = encoder_gen((img_width, img_height), model_config["encoder"])

    # Construct VAE Decoder 
    vae_decoder = decoder_gen(
        (img_width, img_height),  
        model_config["decoder"],
        encoder_result.shape_before_flattening
    )

    z = encoder_result.vae_encoder(encoder_result.inputs)
    x_recon = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_recon])

    # load weights from file
    vae.load_weights('./models/model_{}.th'.format(args.id))
    print("weights loaded")

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    # get side by side plots of original vs. reconstructed
    sample_reconstructions(vae, train_data, test_data, args.id, dataset_max, dataset_min)
    # sample_latent_space(encoder_result.vae_encoder, train_data, args.id, dataset_min, dataset_max)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the id of the config file to use to train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()