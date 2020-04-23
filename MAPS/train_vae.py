import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.utils import plot_model

TRAIN_DATA_PATH = './data/Space_Time_W_Training.npy'
TEST_DATA_PATH = './data/Space_Time_W_Test.npy'

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        """
        TODO 
        """
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var/2) + mean

def kl_reconstruction_loss(true, pred):
    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    # KL divergence loss
    kl_loss = 1 + codings_log_var - K.square(codings_mean) - K.exp(codings_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # Total loss = 50% rec + 50% KL divergence loss
    # On ruihan's advice, I will weight KL down
    kl_weight = 0.1
    return K.mean(reconstruction_loss + kl_weight*kl_loss)
    # return K.mean(reconstruction_loss + kl_loss)

def kl(true, pred):
    kl_loss = 1 + codings_log_var - K.square(codings_mean) - K.exp(codings_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss

def reconstruction(true, pred):
    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    return reconstruction_loss

def encoder_gen(
    input_shape: tuple, 
    filters: int, 
    kernel_size: int, 
    conv_stride: int,
    pool_stride: int,
    activation: str,
    dense_dim: int,    
    latent_dim: int
):
    """
    TODO 
    """

    class EncoderConfig():
        pass 

    encoder_config = EncoderConfig()

    # Construct VAE Encoder layers
    inputs = keras.layers.Input(shape=[width, height, 1])
    z = keras.layers.convolutional.Conv2D(filters, kernel_size, padding='same', activation=activation, strides=stride)(inputs)
    z = keras.layers.MaxPooling2D(pool_size, pool_stride, padding='same')
    z = keras.layers.convolutional.Conv2D(filters, kernel_size, padding='same', activation=activation, strides=stride)(z)
    z = keras.layers.MaxPooling2D(pool_size, pool_stride, padding='same')
    z = keras.layers.convolutional.Conv2D(filters, kernel_size, padding='same', activation=activation, strides=stride)(z)
    z = keras.layers.MaxPooling2D(pool_size, pool_stride, padding='same')
    
    shape_before_flattening = K.int_shape(z) 

    z = keras.layers.Flatten()(inputs)
    z = keras.layers.Dense(dense_dim, activation=activation)(z)

    # Compute mean and log variance 
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(z)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(z)

    z = Sampling()([z_mean, z_log_var])

    # Instantiate Keras model for VAE encoder 
    vae_encoder = keras.Model(inputs = [inputs], outputs=[z_mean, z_log_var, z])

    # Package up everything for the encoder
    encoder_config.inputs = inputs
    encoder_config.z_mean = z_mean
    encoder_config.z_log_var = z_log_var
    encoder_config.z = z
    encoder_config.vae_encoder = vae_encoder 
    encoder_config.shape_before_flattening = shape_before_flattening

    return encoder_config

def decoder_gen(
    latent_dim: int, 
    filters: int, 
    kernel_size: int, 
    conv_stride: int,
    activation: str, 
    width: int, 
    height: int, 
    small_dim: int, 
    shape_before_flat: tuple
):
    """
    TODO 
    """
    decoder_inputs = keras.layers.Input(shape=[latent_dim])
    x = keras.layers.Dense(np.prod(shape_before_flat[1:]), activation=activation)(decoder_inputs)
    x = keras.layers.Reshape(shape_before_flat[1:])(x)
    #outputs = keras.layers.Reshape([width,height])(x)
    #x = keras.layers.Reshape([width,height, 1])(x)
    x = keras.layers.Conv2DTranspose(filters, kernel_size, padding='same', activation=activation, strides=(stride, stride))(x)
    x = keras.layers.Conv2DTranspose(small_filter, kernel_size, padding='same', activation='sigmoid')(x)
    outputs = keras.layers.Reshape([width, height, 1])(x)

    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[outputs])

    return outputs, variational_decoder


def main():
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)

    img_width = train_data.shape[1]
    img_height = train_data.shape[2]

    # network parameters
    latent_dim = 2
    batch_size = 128
    filters = 32
    kernel_size = 3
    dense_dim_1 = 100
    dense_dim_2 = 150
    conv_stride = 1
    pool_stride = 2
    activation = 'relu'
    
    # Construct VAE Encoder 
    encoder_config = encoder_gen(
        (img_width, img_height),
        filters, 
        kernel_size,
        conv_stride, 
        pool_stride, 
        activation, 
        dense_dim,
        latent_dim
    )

    small_filter = 1

    # Construct VAE Decoder 
    outputs, variational_decoder = decoder_gen(
        latent_dim,
        dense_dim_1, 
        dense_dim_2, 
        conv_stride, 
        activation, 
        img_width, 
        img_height, 
        small_filter, 
        shape_before_flattening
    )

    _, _, z = variational_encoder(inputs)
    reconstructions = variational_decoder(z)
    vae = keras.Model(inputs=[inputs], outputs=[reconstructions])

    #latent_loss = -0.5*K.sum(1+codings_log_var - K.exp(codings_log_var) - K.square(codings_mean), axis = -1)
    #variational_ae.add_loss(K.mean(latent_loss)/784.)
    variational_ae.compile(loss=kl_reconstruction_loss, optimizer ="rmsprop", metrics=[kl, reconstruction])
    variational_ae.summary()

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    h = variational_ae.fit(train_data, train_data, epochs=60, batch_size=batch_size, validation_data=[test_data, test_data])

    hdict1 = h.history
    train_loss_values1 = hdict1['loss']
    valid_loss_values1 = hdict1['val_loss']
    epochs1 = range(1, len(train_loss_values1) + 1)
    plt.plot(epochs1, train_loss_values1, 'bo', label='Train loss')
    plt.plot(epochs1, valid_loss_values1, 'b', label='Valid loss')
    plt.title('VAE Training and validation loss')
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    #plt.yscale('log')
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()