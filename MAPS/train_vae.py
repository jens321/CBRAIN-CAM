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
    # NOTE: Why is the image width and height being multiplied here?
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    print("reconstruction loss", reconstruction_loss)
    # KL divergence loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # Total loss = 50% rec + 50% KL divergence loss
    # On ruihan's advice, I will weight KL down
    kl_weight = 0.1
    return K.mean(reconstruction_loss + kl_weight*kl_loss)
    # return K.mean(reconstruction_loss + kl_loss)

def kl(true, pred):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss

def reconstruction(true, pred):
    # Reconstruction loss
    # NOTE: Why is the image width and height being multiplied here?
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    print("reconstruction loss", reconstruction_loss)
    return reconstruction_loss

def encoder_gen(
    input_shape: tuple, 
    filters: int, 
    kernel_size: int, 
    pool_size: int,
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
    inputs = keras.layers.Input(shape=[input_shape[0], input_shape[1], 1])
    zero_padded_inputs = keras.layers.ZeroPadding2D(padding=(1, 0))(inputs)
    print("shape of input after padding", inputs.shape)
    z = keras.layers.convolutional.Conv2D(filters, kernel_size, padding='same', activation=activation, strides=conv_stride)(zero_padded_inputs)
    print("shape after first convolutional layer", z.shape)
    z = keras.layers.MaxPooling2D(pool_size, pool_stride, padding='same')(z)
    print("shape after first pooling layer", z.shape)
    z = keras.layers.convolutional.Conv2D(filters, kernel_size, padding='same', activation=activation, strides=conv_stride)(z)
    print("shape after second convolutional layer", z.shape)
    z = keras.layers.MaxPooling2D(pool_size, pool_stride, padding='same')(z)
    print("shape after second pooling layer", z.shape)
    z = keras.layers.convolutional.Conv2D(filters, kernel_size, padding='same', activation=activation, strides=conv_stride)(z)
    print("shape after third convolutional layer", z.shape)
    z = keras.layers.MaxPooling2D(pool_size, pool_stride, padding='same')(z)
    print("shape after third pooling layer", z.shape)
    
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
    original_input: tuple,
    shape_before_flat: tuple
):
    """
    TODO 
    """
    decoder_inputs = keras.layers.Input(shape=[latent_dim])
    # Map latent space to input size after last pooling layer of the encoder 
    x = keras.layers.Dense(np.prod(shape_before_flat[1:]), activation=activation)(decoder_inputs)

    # Reshape input to be an image 
    x = keras.layers.Reshape(shape_before_flat[1:])(x)

    # Start tranpose convolutional layers that upsample the image
    print("shape at beginning of decoder", x.shape)
    x = keras.layers.Conv2DTranspose(filters, kernel_size, padding='same', activation=activation, strides=conv_stride)(x)
    print("shape after first convolutional transpose layer", x._keras_shape)
    x = keras.layers.Conv2DTranspose(filters, kernel_size, padding='same', activation=activation, strides=conv_stride)(x)
    print("shape after second convolutional transpose layer", x._keras_shape)
    x = keras.layers.Conv2DTranspose(filters, kernel_size, padding='same', activation=activation, strides=conv_stride)(x)
    print("shape after third convolutional transpose layer", x._keras_shape)

    x = keras.layers.Conv2DTranspose(1, kernel_size, padding='same', activation='sigmoid')(x)
    print("shape after special convolutional transpose layer", x._keras_shape)

    cropped_outputs = keras.layers.Cropping2D(cropping=(1, 0))(x)
    print("shape after cropping", cropped_outputs._keras_shape)

    outputs = keras.layers.Reshape([original_input[0], original_input[1], 1])(cropped_outputs)
    print("shape after reshaping", outputs._keras_shape)

    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[outputs])

    return outputs, variational_decoder


def main():

    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)

    img_width = train_data.shape[1]
    img_height = train_data.shape[2]

    print("Image shape:", img_width, img_height)

    # network parameters
    latent_dim = 2
    batch_size = 128
    filters = 32
    kernel_size = 3
    pool_size = 2
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
        pool_size,
        conv_stride, 
        pool_stride, 
        activation, 
        dense_dim_1,
        latent_dim
    )

    small_filter = 1
    conv_stride = 2

    # Construct VAE Decoder 
    outputs, vae_decoder = decoder_gen(
        latent_dim,
        filters,
        kernel_size,
        conv_stride, 
        activation, 
        (img_width, img_height),  
        encoder_config.shape_before_flattening
    )

    _, _, z = encoder_config.vae_encoder(encoder_config.inputs)
    reconstructions = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_config.inputs], outputs=[reconstructions])

    # latent_loss = -0.5*K.sum(1+codings_log_var - K.exp(codings_log_var) - K.square(codings_mean), axis = -1)
    # variational_ae.add_loss(K.mean(latent_loss)/784.)
    vae.compile(loss=reconstruction, optimizer ="rmsprop", metrics=[reconstruction])
    vae.summary()

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    h = vae.fit(train_data, train_data, epochs=60, batch_size=batch_size, validation_data=[test_data, test_data])

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