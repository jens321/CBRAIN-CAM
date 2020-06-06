import argparse 
import json 

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import backend as K
import tensorflow as tf 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE

from train_vae import encoder_gen, decoder_gen


def reconstruct_targets(vae, test_data, targets, id, dataset_max, dataset_min):
    """
    TODO
    """
    original_samples = []
    recon_samples = []

    min_max = []

    for target in targets:

        sample = test_data[target]
        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :128*30]
        sample_log_var = sample_mean_var[0, 128*30:]

        recon_sample = np.random.multivariate_normal(sample_mean, np.exp(sample_log_var) * np.identity(128*30))
        
        sample = np.interp(sample, (0, 1), (dataset_min, dataset_max))
        recon_sample = np.interp(recon_sample, (0, 1), (dataset_min, dataset_max))

        print("original sample", sample.reshape((128*30,)))
        print("reconstructed sample", recon_sample)
        print(np.max(np.abs(sample.reshape((128*30,)) - recon_sample)))
        max_reconstructed = np.max(np.abs(recon_sample))
        print("max of reconstructed", max_reconstructed)
        max_sample = np.max(sample.reshape((128*30,)))
        print("max of original", max_sample)
        min_reconstructed = np.min(recon_sample)
        print("min of reconstructed", min_reconstructed)
        min_sample = np.min(sample.reshape((128*30,)))
        print("min of original", min_sample)
        recon_sample = recon_sample.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recon_samples.append(recon_sample)

        min_max.append((min(min_reconstructed, min_sample), max(max_reconstructed, max_sample)))

    fig, axs = plt.subplots(5, 2)

    for i in range(5): 
        vmin = min_max[i][0]
        vmax = min_max[i][1]
    
        sub_img = axs[i, 0].imshow(original_samples[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        axs[i, 0].set_ylim(axs[i, 0].get_ylim()[::-1])
        fig.colorbar(sub_img, ax=axs[i, 0])
        # ax.set_title("Original Sample")

        sub_img = axs[i, 1].imshow(recon_samples[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        axs[i, 1].set_ylim(axs[i, 1].get_ylim()[::-1])
        fig.colorbar(sub_img, ax=axs[i, 1])
        # ax.set_title("Reconstructed Sample")

    plt.savefig('./model_graphs/target_test_reconstructions_with_var_{}.png'.format(id))



def sample_reconstructions(vae, train_data, test_data, id, dataset_max, dataset_min): 
    """
    TODO 
    """

    # get random sample 
    original_samples = []
    recon_samples = []

    min_max = []

    for i in range(5):
        rand_sample = np.random.randint(0, len(train_data))

        sample = train_data[rand_sample]
        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :128*30]
        sample_log_var = sample_mean_var[0, 128*30:]

        recon_sample = np.random.multivariate_normal(sample_mean, np.exp(sample_log_var) * np.identity(128*30))
        
        sample = np.interp(sample, (0, 1), (dataset_min, dataset_max))
        recon_sample = np.interp(recon_sample, (0, 1), (dataset_min, dataset_max))

        print("original sample", sample.reshape((128*30,)))
        print("reconstructed sample", recon_sample)
        print(np.max(np.abs(sample.reshape((128*30,)) - recon_sample)))
        max_reconstructed = np.max(np.abs(recon_sample))
        print("max of reconstructed", max_reconstructed)
        max_sample = np.max(sample.reshape((128*30,)))
        print("max of original", max_sample)
        min_reconstructed = np.min(recon_sample)
        print("min of reconstructed", min_reconstructed)
        min_sample = np.min(sample.reshape((128*30,)))
        print("min of original", min_sample)
        recon_sample = recon_sample.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recon_samples.append(recon_sample)

        min_max.append((min(min_reconstructed, min_sample), max(max_reconstructed, max_sample)))

    fig, axs = plt.subplots(5, 2)

    for i in range(5): 
        vmin = min_max[i][0]
        vmax = min_max[i][1]
    
        sub_img = axs[i, 0].imshow(original_samples[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        axs[i, 0].set_ylim(axs[i, 0].get_ylim()[::-1])
        fig.colorbar(sub_img, ax=axs[i, 0])
        # ax.set_title("Original Sample")

        sub_img = axs[i, 1].imshow(recon_samples[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        axs[i, 1].set_ylim(axs[i, 1].get_ylim()[::-1])
        fig.colorbar(sub_img, ax=axs[i, 1])
        # ax.set_title("Reconstructed Sample")

    plt.savefig('./model_graphs/reconstructed_train_samples_with_var{}.png'.format(id))

def sample_latent_space(vae_encoder, train_data, test_data, id, dataset_min, dataset_max, test_labels): 
    """
    TODO 
    """

    # Predict latent train & test data
    _, _, z_test = vae_encoder.predict(test_data)
    _, _, z_train = vae_encoder.predict(train_data)

    # Apply scaling and tsne 
    sc = StandardScaler()
    z_train_std = sc.fit_transform(z_train)
    
    z_test_std = sc.transform(z_test)

    # Instantiate PCA 
    pca = PCA(n_components=32)
    pca.fit(z_train_std)

    z_test_pca = pca.transform(z_test_std)

    # Instantiate TSNE
    tsne = TSNE(n_components=2)

    z_test_tsne = tsne.fit_transform(z_test_pca)

    # Make plot of latent test data 
    plt.scatter(x=z_test_tsne[:, 0], y=z_test_tsne[:, 1], c=test_labels)
    plt.colorbar()

    plt.savefig('./model_graphs/latent_space_with_pca_{}.png'.format(id))

def generate_samples(decoder, dataset_min, dataset_max, latent_dim: int, id):
    """
    Sample points from prior and send through decoder to get 
    sample images.
    """
    # sample from prior 
    num_samples = 5
    z = np.random.normal(size=(num_samples, latent_dim))

    # Get output from decoder 
    sample_mean_var = decoder.predict(z)

    # Extract mean and variance 
    sample_mean = sample_mean_var[:, :128*30]
    sample_log_var = sample_mean_var[:, 128*30:]

    fig, axs = plt.subplots(5, 1)

    recon_samples = []
    for i in range(num_samples):
        print(sample_mean[i])
        print(sample_mean[i].shape)
        # Sample from gaussian decoder outputs 
        recon_sample = np.random.multivariate_normal(sample_mean[i], np.exp(sample_log_var[i]) * np.identity(128*30))

        # Unnormalize sample 
        recon_sample = np.interp(recon_sample, (0, 1), (dataset_min, dataset_max))

        # Reshape
        recon_sample = recon_sample.reshape((30, 128))

        recon_samples.append(recon_sample)

    vmin = np.min(recon_samples)
    vmax = np.max(recon_samples)
    for i in range(num_samples):
        # Show image
        sub_img = axs[i].imshow(recon_sample, cmap='coolwarm', vmin=vmin, vmax=vmax)

        # Flip y-axis
        axs[i].set_ylim(axs[i].get_ylim()[::-1])
        
    # fig.colorbar(sub_img, ax=axs)
    plt.tight_layout()
    plt.savefig('./model_graphs/generated_samples_{}.png'.format(id))



def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/config_{}.json".format(args.id))
    model_config = json.load(f)
    f.close()

    train_data = np.load(model_config["data"]["training_data_path"])
    test_data = np.load(model_config["data"]["test_data_path"])

    # test_labels = np.load(model_config["data"]["test_labels"])[:, 0, 0]
    test_labels = np.load(model_config["data"]["test_labels"])
    print("Test labels shape:", test_labels.shape, model_config["data"]["test_labels"])

    dataset_max = np.load(model_config["data"]["max_scalar"])
    dataset_min = np.load(model_config["data"]["min_scalar"])

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

    _, _, z = encoder_result.vae_encoder(encoder_result.inputs)
    x_mu_var = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_mu_var])

    # load weights from file
    vae.load_weights('./models/model_{}.th'.format(args.id))
    print("weights loaded")

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    # get side by side plots of original vs. reconstructed
    # sample_reconstructions(vae, train_data, test_data, args.id, dataset_max, dataset_min)
    # reconstruct_targets(vae, test_data, [2, 15, 66 , 85, 94], args.id, dataset_max, dataset_min)
    sample_latent_space(encoder_result.vae_encoder, train_data, test_data, args.id, dataset_min, dataset_max, test_labels)
    # generate_samples(vae_decoder, dataset_min, dataset_max, model_config["encoder"]["latent_dim"], args.id)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the id of the config file to use to train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()
