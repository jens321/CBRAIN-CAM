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

def loss(true, pred):
    return mse(true, pred)

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
        tuple(encoder_config["conv_1"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_1"]["stride"]
    )(zero_padded_inputs)
    print("shape after first convolutional layer", z.shape)

    # z = keras.layers.AveragePooling2D(
    #     encoder_config["avg_pool_1"]["pool_size"],
    #     encoder_config["avg_pool_1"]["pool_stride"],
    #     padding="same"
    # )(z)
    # print("shape after first pooling layer", z.shape)

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

    # z = keras.layers.AveragePooling2D(
    #     encoder_config["avg_pool_2"]["pool_size"],
    #     encoder_config["avg_pool_2"]["pool_stride"],
    #     padding="same"
    # )(z)
    # print("shape after second pooling layer", z.shape)

    shape_before_flattening = K.int_shape(z) 

    z = keras.layers.Flatten()(z)

    # Compute final latent state 
    z = keras.layers.Dense(encoder_config["latent_dim"], name='z_mean')(z)

    # Instantiate Keras model for VAE encoder 
    vae_encoder = keras.Model(inputs = [inputs], outputs=[z])

    # Package up everything for the encoder
    encoder_result.inputs = inputs
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
    # x = keras.layers.Dense(np.prod(shape_before_flat[1:]), activation=decoder_config["activation"])(decoder_inputs)
    x = keras.layers.Dense(2048)(decoder_inputs)

    # Reshape input to be an image 
    # x = keras.layers.Reshape(shape_before_flat[1:])(x)
    x = keras.layers.Reshape((4, 16, 32))(x)

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
    print("shape after second convolutional layer", x._keras_shape)

    x_recon = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_3"]["filter_num"], 
        tuple(decoder_config["conv_t_3"]["kernel_size"]), 
        padding='same', 
        strides=decoder_config["conv_t_3"]["stride"],
        activation=decoder_config["conv_t_3"]["activation"]
    )(x)
    print("shape after conv recon layer", x_recon._keras_shape)

    x_recon = keras.layers.Cropping2D(cropping=(1, 0))(x_recon)
    print("shape after cropping", x_recon._keras_shape)

    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[x_recon])

    return variational_decoder

def plot_training_losses(h, id):
    """
    TODO 
    """
    hdict = h.history
    print(hdict)

    train_losses = hdict['loss']
    valid_losses = hdict['val_loss']

    epochs = range(1, len(train_losses) + 1)

    # Plot combined loss 
    plt.plot(epochs, train_losses, 'b', label='Train')
    plt.plot(epochs, valid_losses, 'r', label='Valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(prop={'size': 10})
    plt.title("Combined Loss")

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

    z = encoder_result.vae_encoder(encoder_result.inputs)
    x_recon = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_recon])

    # Specify the optimizer 
    optimizer = keras.optimizers.Adam(lr=model_config['optimizer']['lr'])

    # Compile model 
    vae.compile(loss='mse', optimizer=optimizer, metrics=[loss])
    vae.summary()

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    print("train data shape", train_data.shape)
    print("test data shape", test_data.shape)

    checkpoint = ModelCheckpoint(
        './models/model_{}.th'.format(args.id), 
        monitor='loss', 
        verbose=1,
        save_best_only=True ,
        save_weights_only=True
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
