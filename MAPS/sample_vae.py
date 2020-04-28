import argparse 
import json 

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import backend as K

from train_vae import encoder_gen, decoder_gen

def sample_reconstructions(vae, train_data, test_data, id): 
    """
    TODO 
    """

    # get random sample 
    original_samples = []
    recon_samples = []
    
    for i in range(5):
        rand_sample = np.random.randint(0, len(train_data))

        sample = train_data[rand_sample]
        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :128*30]
        sample_log_var = sample_mean_var[0, 128*30:]
        recon_sample = np.random.multivariate_normal(sample_mean, np.exp(sample_log_var) * np.identity(128*30))
        recon_sample = recon_sample.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recon_samples.append(recon_sample)

    fig, axs = plt.subplots(5, 2)

    for i in range(10): 
        print(i)
        if i % 2 == 0:
            axs[int(i/2), 0].imshow(original_samples[int(i/2)], cmap='coolwarm')
            # ax.set_title("Original Sample")
        elif i % 2 == 1:
            axs[int(i/2), 1].imshow(recon_samples[int(i/2)], cmap='coolwarm')
            # ax.set_title("Reconstructed Sample")

    plt.savefig('./model_graphs/reconstructed_train_samples_{}.png'.format(id))

def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/config_{}.json".format(args.id))
    model_config = json.load(f)
    f.close()

    train_data = np.load(model_config["data"]["training_data_path"])
    test_data = np.load(model_config["data"]["test_data_path"])

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
    sample_reconstructions(vae, train_data, test_data, args.id)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the id of the config file to use to train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()