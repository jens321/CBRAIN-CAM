import math
import os 
import argparse
import json 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy, mse
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets.cifar10 import load_data

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        """
        TODO 
        """
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var/2) + mean

def kl_reconstruction_loss(z_log_var, z_mean, kl_weight):
    def _kl_reconstruction_loss(true, pred):
        """
        Compute both KL term and reconstruction term of the VAE loss and combine
        into ELBO.  
        """
        input_dim = 32 * 32 * 3

        print("shape of true", true._keras_shape)
        print("shape of pred", pred._keras_shape)
        true = tf.reshape(true, [-1, input_dim])

        x_mu = pred[:, :input_dim]

        # x_var = 0.05 * K.ones_like(x_mu)
        x_log_var = pred[:, input_dim:]

        # Gaussian reconstruction loss

        # mse = -0.5 * K.sum(K.square(true - x_mu)/x_var, axis=1)
        mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)

        # var_trace = -0.5 * K.sum(K.log(x_var), axis=1)
        var_trace = -0.5 * K.sum(x_log_var, axis=1)

        log2pi = -0.5 * input_dim * np.log(2 * np.pi)
        
        log_likelihood = mse + var_trace + log2pi
        print("log likelihood shape", log_likelihood.shape)

        # NOTE: We don't take a mean here, since we first want to add the KL term
        reconstruction_loss = -log_likelihood

        # KL divergence loss
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=1)
        kl_loss *= -0.5

        # Total loss = 50% rec + 50% KL divergence loss
        # NOTE: On ruihan's advice, I will weight KL down

        return K.mean(reconstruction_loss + kl_weight * kl_loss)

    return _kl_reconstruction_loss

def kl(z_log_var, z_mean):
    def _kl(true, pred):
        """
        Compute the KL term of the VAE loss. 
        """
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # kl_loss = K.print_tensor(kl_loss, message='EULA PEULA')
        return K.mean(kl_loss)
    
    return _kl

def reconstruction(true, pred):
    """
    Compute the reconstruction term of the VAE loss based
    on Gaussian decoder. 
    """
    input_dim = 32 * 32 * 3

    print("shape of true", true._keras_shape)
    print("shape of pred", pred._keras_shape)
    true = tf.reshape(true, [-1, input_dim])

    x_mu = pred[:, :input_dim]

    # x_var = 0.05 * K.ones_like(x_mu)
    x_log_var = pred[:, input_dim:]

    # Gaussian reconstruction loss

    # mse = -0.5 * K.sum(K.square(true - x_mu)/x_var, axis=1)
    mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)

    # var_trace = -0.5 * K.sum(K.log(x_var), axis=1)
    var_trace = -0.5 * K.sum(x_log_var, axis=1)

    log2pi = -0.5 * input_dim * np.log(2 * np.pi)
    
    log_likelihood = mse + var_trace + log2pi
    print("log likelihood shape", log_likelihood.shape)

    return K.mean(-log_likelihood)

def encoder_gen(input_shape: tuple, encoder_config: dict):
    """
    Create the architecture for the VAE encoder. 
    """

    class EncoderResult():
        pass 

    encoder_result = EncoderResult()

    # Construct VAE Encoder layers
    inputs = keras.layers.Input(shape=[input_shape[0], input_shape[1], input_shape[2]])

    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_1"]["filter_num"], 
        tuple(encoder_config["conv_1"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_1"]["stride"]
    )(inputs)
    print("shape after first convolutional layer", z.shape)

    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_2"]["filter_num"], 
        tuple(encoder_config["conv_2"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_2"]["stride"]
    )(z)

    print("shape after second convolutional layer", z.shape)

    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_3"]["filter_num"], 
        tuple(encoder_config["conv_3"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_3"]["stride"]
    )(z)

    print("shape after third convolutional layer", z.shape)

    shape_before_flattening = K.int_shape(z) 

    z = keras.layers.Flatten()(z)

    # Compute mean and log variance 
    z_mean = keras.layers.Dense(encoder_config["latent_dim"], name='z_mean')(z)
    z_log_var = keras.layers.Dense(encoder_config["latent_dim"], name='z_log_var')(z)

    z = Sampling()([z_mean, z_log_var])

    # Instantiate Keras model for VAE encoder 
    vae_encoder = keras.Model(inputs = [inputs], outputs=[z_mean, z_log_var, z])

    # Package up everything for the encoder
    encoder_result.inputs = inputs
    encoder_result.z_mean = z_mean
    encoder_result.z_log_var = z_log_var
    encoder_result.z = z
    encoder_result.vae_encoder = vae_encoder 
    encoder_result.shape_before_flattening = shape_before_flattening

    return encoder_result

def decoder_gen(
    original_input: tuple,
    decoder_config: dict, 
    shape_before_flat: tuple
):
    """
    Create the architecture for the VAE decoder 
    """
    decoder_inputs = keras.layers.Input(shape=[decoder_config["latent_dim"]])
    # Map latent space to input size after last pooling layer of the encoder 
    x = keras.layers.Dense(2048)(decoder_inputs)

    # Reshape input to be an image 
    x = keras.layers.Reshape((4, 4, 128))(x)

    # Start tranpose convolutional layers that upsample the image
    print("shape at beginning of decoder", x.shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_1"]["filter_num"], 
        tuple(decoder_config["conv_t_1"]["kernel_size"]), 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_1"]["stride"]
    )(x)
    print("shape after first convolutional transpose layer", x._keras_shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_2"]["filter_num"], 
        tuple(decoder_config["conv_t_2"]["kernel_size"]), 
        padding='same', 
        strides=decoder_config["conv_t_2"]["stride"],
        activation=decoder_config["activation"]
    )(x)
    print("shape after second convolutional transpose layer", x._keras_shape)

    x_mu = keras.layers.Conv2DTranspose(
        decoder_config["conv_mu"]["filter_num"], 
        tuple(decoder_config["conv_mu"]["kernel_size"]), 
        padding='same',  
        strides=decoder_config["conv_mu"]["stride"],
        activation=decoder_config["conv_mu"]["activation"]
    )(x)
    print("shape after convolutional mu layer", x_mu._keras_shape)

    x_log_var = keras.layers.Conv2DTranspose(
        decoder_config["conv_log_var"]["filter_num"], 
        tuple(decoder_config["conv_log_var"]["kernel_size"]), 
        padding='same',  
        strides=decoder_config["conv_log_var"]["stride"],
        activation=decoder_config["conv_log_var"]["activation"]
    )(x)
    print("shape after convolutional log var layer", x_log_var._keras_shape)

    x_mu = keras.layers.Flatten()(x_mu)
    x_log_var = keras.layers.Flatten()(x_log_var)

    # variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[x_mu])
    x_mu_log_var = keras.layers.Concatenate(axis=1)([x_mu, x_log_var])

    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[x_mu_log_var])

    return variational_decoder

def plot_training_losses(h, config_file: str):
    """
    Plot training loss graphs for 
        (1) KL term
        (2) Reconstruction term
        (3) Total ELBO loss  
    """
    hdict = h.history
    print(hdict)

    train_reconstruction_losses = hdict['reconstruction']
    valid_reconstruction_losses = hdict['val_reconstruction']

    kl_train_losses = hdict['_kl']
    kl_valid_losses = hdict['val__kl']

    total_train_losses = hdict['_kl_reconstruction_loss']
    total_valid_losses = hdict['val__kl_reconstruction_loss']

    epochs = range(1, len(train_reconstruction_losses) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.8, 4.8))

    # Plot combined loss 
    ax1.plot(epochs, total_train_losses, 'b', label='Train')
    ax1.plot(epochs, total_valid_losses, 'r', label='Valid')
    ax1.set(xlabel="Epochs", ylabel="Loss")
    ax1.legend(prop={'size': 10})
    ax1.set_title("Combined Loss")

    # Plot KL 
    ax2.plot(epochs, kl_train_losses, 'b', label='Train')
    ax2.plot(epochs, kl_valid_losses, 'r', label='Valid')
    ax2.set(xlabel="Epochs", ylabel="Loss")
    ax2.legend(prop={'size': 10})
    ax2.set_title("KL Loss")

    # Plot reconstruction loss 
    ax3.plot(epochs, train_reconstruction_losses, 'b', label='Train')
    ax3.plot(epochs, valid_reconstruction_losses, 'r', label='Valid')
    ax3.set(xlabel="Epochs", ylabel="Loss")
    ax3.legend(prop={'size': 10})
    ax3.set_title("Reconstruction Loss")
    
    plt.tight_layout()

    plt.savefig('./model_graphs/model_losses_{}.png'.format(config_file))

def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/{}.json".format(args.config_file))
    model_config = json.load(f)
    f.close()

    (train_data, _), (test_data, _) = load_data()

    assert train_data.shape[1:] == (32, 32, 3)
    assert test_data.shape[1:] == (32, 32, 3)

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
    x_mu_log_var = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_mu_log_var])

    # Specify the optimizer 
    optimizer = keras.optimizers.Adam(lr=model_config['optimizer']['lr'])

    # Compile model 
    vae.compile(
        # loss=reconstruction, 
        loss=kl_reconstruction_loss(encoder_result.z_log_var, encoder_result.z_mean, model_config["kl_weight"]), 
        optimizer=optimizer, 
        metrics=[
            reconstruction, 
            kl(encoder_result.z_log_var, encoder_result.z_mean), 
            kl_reconstruction_loss(encoder_result.z_log_var, encoder_result.z_mean, model_config["kl_weight"])
        ]
    )
    vae.summary()

    print("train data shape", train_data.shape)
    print("test data shape", test_data.shape)

    checkpoint = ModelCheckpoint(
        './models/model_{}.th'.format(args.config_file), 
        monitor='val_loss', 
        verbose=1,
        save_best_only=True,
        save_weights_only=False # since we're playing with different architectures, this may be better
    )
    callbacks_list = [checkpoint]

    h = vae.fit(
        x=train_data, 
        y=train_data, 
        epochs=model_config["train_epochs"], 
        batch_size=model_config["batch_size"], 
        validation_data=[test_data, test_data],
        callbacks=callbacks_list
    )

    plot_training_losses(h, args.config_file)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='This option specifies the config file to use to construct and train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()
