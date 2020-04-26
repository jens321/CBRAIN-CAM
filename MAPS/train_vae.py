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

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        """
        TODO 
        """
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var/2) + mean

def kl_reconstruction_loss(z_log_var, z_mean):
    def _kl_reconstruction_loss(true, pred):
        """
        TODO 
        """
        true = tf.reshape(true, [-1, 128 * 30])

        x_mu = pred[:, :128*30]
        x_var = pred[:, 128*30:]

        # Gaussian reconstruction loss
        mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_var), axis=1)
        var_trace = -0.5 * K.sum(x_var, axis=1)
        log2pi = -0.5 * 128 * 30 * np.log(2 * np.pi)
        
        log_likelihood = mse + var_trace + log2pi
        print("log likelihood shape", log_likelihood.shape)

        reconstruction_loss = K.mean(-log_likelihood)

        # KL divergence loss
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Total loss = 50% rec + 50% KL divergence loss
        # NOTE: On ruihan's advice, I will weight KL down
        # kl_weight = 0.1
        # return K.mean(reconstruction_loss + kl_weight * kl_loss)

        return K.mean(reconstruction_loss + kl_loss)

    return _kl_reconstruction_loss

def kl(z_log_var, z_mean):
    def _kl(true, pred):
        """
        TODO 
        """
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss
    
    return _kl

def reconstruction(true, pred):
    """
    TODO
    """
    true = tf.reshape(true, [-1, 128 * 30])

    x_mu = pred[:, :128*30]
    x_log_var = pred[:, 128*30:]

    mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)
    var_trace = -0.5 * K.sum(x_log_var, axis=1)
    log2pi = -0.5 * 128 * 30 * np.log(2 * np.pi)
    
    log_likelihood = mse + var_trace + log2pi
    print("log likelihood shape", log_likelihood.shape)

    return K.mean(-log_likelihood)

def encoder_gen(input_shape: tuple, encoder_config: dict):
    """
    TODO 
    """

    class EncoderResult():
        pass 

    encoder_result = EncoderResult()

    # Construct VAE Encoder layers
    inputs = keras.layers.Input(shape=[input_shape[0], input_shape[1], 1])
    zero_padded_inputs = keras.layers.ZeroPadding2D(padding=(1, 0))(inputs)

    print("shape of input after padding", inputs.shape)
    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_1"]["filter_num"], 
        encoder_config["conv_1"]["kernel_size"], 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_1"]["stride"]
    )(zero_padded_inputs)

    print("shape after first convolutional layer", z.shape)
    z = keras.layers.MaxPooling2D(
        encoder_config["max_pool_1"]["pool_size"], 
        encoder_config["max_pool_1"]["pool_stride"], 
        padding='same'
    )(z)

    print("shape after first pooling layer", z.shape)
    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_2"]["filter_num"], 
        encoder_config["conv_2"]["kernel_size"], 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_2"]["stride"]
    )(zero_padded_inputs)

    print("shape after second convolutional layer", z.shape)
    z = keras.layers.MaxPooling2D(
        encoder_config["max_pool_2"]["pool_size"], 
        encoder_config["max_pool_2"]["pool_stride"], 
        padding='same'
    )(z)

    print("shape after second pooling layer", z.shape)
    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_3"]["filter_num"], 
        encoder_config["conv_3"]["kernel_size"], 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_3"]["stride"]
    )(zero_padded_inputs)

    print("shape after third convolutional layer", z.shape)
    z = keras.layers.MaxPooling2D(
        encoder_config["max_pool_3"]["pool_size"], 
        encoder_config["max_pool_3"]["pool_stride"], 
        padding='same'
    )(z)
    print("shape after third pooling layer", z.shape)
    
    shape_before_flattening = K.int_shape(z) 

    z = keras.layers.Flatten()(inputs)
    z = keras.layers.Dense(encoder_config["dense_1"]["dim"], activation=encoder_config["activation"])(z)

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
    TODO 
    """
    decoder_inputs = keras.layers.Input(shape=[decoder_config["latent_dim"]])
    # Map latent space to input size after last pooling layer of the encoder 
    x = keras.layers.Dense(np.prod(shape_before_flat[1:]), activation=decoder_config["activation"])(decoder_inputs)

    # Reshape input to be an image 
    x = keras.layers.Reshape(shape_before_flat[1:])(x)

    # Start tranpose convolutional layers that upsample the image
    print("shape at beginning of decoder", x.shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_1"]["filter_num"], 
        decoder_config["conv_t_1"]["kernel_size"], 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_1"]["stride"]
    )(x)
    print("shape after first convolutional transpose layer", x._keras_shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_2"]["filter_num"], 
        decoder_config["conv_t_2"]["kernel_size"], 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_2"]["stride"]
    )(x)
    print("shape after second convolutional transpose layer", x._keras_shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_3"]["filter_num"], 
        decoder_config["conv_t_3"]["kernel_size"], 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_3"]["stride"]
    )(x) 
    print("shape after third convolutional transpose layer", x._keras_shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_4"]["filter_num"], 
        decoder_config["conv_t_4"]["kernel_size"], 
        padding='same', 
        activation=decoder_config["activation"]
    )(x)
    print("shape after fourth convolutional transpose layer", x._keras_shape)

    cropped_outputs = keras.layers.Cropping2D(cropping=(1, 0))(x)
    print("shape after cropping", cropped_outputs._keras_shape)

    x = keras.layers.Flatten()(cropped_outputs)

    x_mu_var = keras.layers.Dense(2 * np.prod(original_input), activation=decoder_config["dense_mu_var"]["activation"])(x)

    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[x_mu_var])

    return variational_decoder

def plot_training_losses(h, id):
    """
    TODO 
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.8, 9.6))

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

    plt.savefig('./model_graphs/model_losses_{}.png'.format(id))

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

    # Specify the optimizer 
    optimizer = keras.optimizers.Adam(lr=model_config['optimizer']['lr'])

    # Compile model 
    vae.compile(
        loss=kl_reconstruction_loss(encoder_result.z_mean, encoder_result.z_log_var), 
        optimizer=optimizer, 
        metrics=[reconstruction, kl(encoder_result.z_mean, encoder_result.z_log_var), kl_reconstruction_loss(encoder_result.z_mean, encoder_result.z_log_var)]
    )
    vae.summary()

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    print("train data shape", train_data.shape)
    print("test data shape", test_data.shape)

    h = vae.fit(
        x=train_data, 
        y=train_data, 
        epochs=model_config["train_epochs"], 
        batch_size=model_config["batch_size"], 
        validation_data=[test_data, test_data]
    )

    vae.save_weights('./models/model_{}.th'.format(args.id))

    plot_training_losses(h, args.id)

    # # plot truth vs predicted 
    # truth_image = train_data[1356]
    # truth_image = truth_image.reshape(1, 30, 128, 1)

    # predicted_image = vae.predict(truth_image)[0, :, :, 0]
    # print("shape of predicted image", predicted_image.shape)
    # truth_image = truth_image[0, :, :, 0]

    # fig = plt.figure()
    # a = fig.add_subplot(1, 2, 1)
    # imgplot = plt.imshow(truth_image, cmap='coolwarm')
    # a.set_title('Truth')
    # # plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    # a = fig.add_subplot(1, 2, 2)
    # imgplot = plt.imshow(predicted_image, cmap='coolwarm')
    # # imgplot.set_clim(0.0, 0.7)
    # a.set_title('Predicted')
    # # plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    # plt.savefig('compared.png')

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the id of the config file to use to train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()
