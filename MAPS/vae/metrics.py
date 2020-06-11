import argparse 
import json 

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import backend as K
import tensorflow as tf 

from train_vae import encoder_gen, decoder_gen

def spectrum_gen(h, dx):
    nx = len(h)

    # Get half the length of the series to avoid redudant information
    npositive = nx//2
    pslice = slice(1, npositive)

    # Get frequencies
    freqs = np.fft.fftfreq(nx, d=dx)[pslice] 

    # Perform the fft 
    ft = np.fft.fft(h)[pslice]

    # Remove imaginary componant of the fft and square
    psraw = np.conjugate(ft) *ft

    # Double to account for the negative half that was removed above
    psraw *= 2.0

    # Normalization for the power spectrum
    psraw /= nx**2

    # Go from the Power Spectrum to Power Density
    psdraw = psraw * dx * nx

    return freqs, psraw, psdraw

def spectrum_generator(targets, features, levels, time_space):
    targ_freqs, targ_psraw, targ_psdraw = spectrum_gen(np.squeeze(targets[1,:]), time_space)
    depth = len(targ_psdraw)
    target_collector = np.zeros(shape=(levels, depth))
    target_collector[:,:] = np.nan
    feature_collector = np.zeros(shape=(levels, depth))
    feature_collector[:,:] = np.nan
    counter = 0
    for i in range(levels):
            target = np.squeeze(targets[i, :])
            feature = np.squeeze(features[i, :])
            targ_freqs, targ_psraw, targ_psdraw = spectrum_gen(target, time_space)
            feat_freqs, feat_psraw, feat_psdraw = spectrum_gen(feature, time_space)
            target_collector[i, :] = targ_psdraw
            feature_collector[i, :] = feat_psdraw
    rep_target = np.nanmean(target_collector, axis = 0)
    rep_pred = np.nanmean(feature_collector, axis = 0)
    #plotter(rep_target, rep_pred, targ_freqs, "Image average Signal")
    #return targ_freqs, feat_freqs, target_collector, feature_collector
    return rep_target, rep_pred, targ_freqs

def mse_metric(p, q):
    mse = np.square(np.subtract(p,q)).mean()
    return mse

def pdf_gen(dist):
    mu, std = norm.fit(dist)
    #print("this is the dist", dist.shape)
    if dist.ndim > 2:
        dist = np.reshape(dist, (len(dist)*len(dist[0]),len(dist[0][0])))
    plt.hist(dist, bins=25, density=True, alpha=0.6)
    #print("Graphed")
    xmin, xmax = plt.xlim()
    #print("limits")
    x = np.linspace(xmin, xmax, len(dist))
    #print("linspace")
    pdf = norm.pdf(x, mu, std)
    #print("made it to pdf func end")
    return pdf

def hellinger(p, q):
    # p = pdf_gen(p)
    # print("sum of pdf", np.sum(p))
    # q = pdf_gen(q)
    p = p/np.sum(p)
    q = q/np.sum(q)
    hd = np.sqrt(np.sum((np.sqrt(p.ravel()) - np.sqrt(q.ravel())) ** 2)) / np.sqrt(2)
    return hd

def compute_metrics(vae, random_vae, train_data, test_data, id, dataset_max, dataset_min):
    hds = []
    hds_random = []

    mses = []
    mses_random = []

    spectrum_pred_random = []
    spectrum_pred = []
    spectrum_truth = []

    original_samples = []
    recon_samples = []

    i = 0
    for sample in test_data:
        print(i)

        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :128*30]
        sample_log_var = sample_mean_var[0, 128*30:]

        sample_mean_var_random = random_vae.predict(np.expand_dims(sample, 0))
        sample_mean_random = sample_mean_var_random[0, :128*30]

        # For the reconstruction, we take just the mean
        recon_sample = sample_mean
        recon_sample = recon_sample.reshape((30, 128))

        recon_sample_random = sample_mean_random
        recon_sample_random = recon_sample_random.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recon_samples.append(recon_sample)

        # Compute hellinger
        h = hellinger(np.array(sample[:, :, 0]), np.array(recon_sample))
        hds.append(h)
        h_random = hellinger(np.array(sample[:, :, 0]), np.array(recon_sample_random))
        hds_random.append(h_random)

        # Compute MSE
        mse = mse_metric(np.array(sample[:, :, 0]), np.array(recon_sample))
        mses.append(mse)
        mse_random = mse_metric(np.array(sample[:, :, 0]), np.array(recon_sample_random))
        mses_random.append(mse_random)

        # Compute spectral 
        rep_target, rep_pred, targ_freqs = spectrum_generator(sample[:, :, 0], recon_sample, 30, 1)
        _, rep_pred_random, _ = spectrum_generator(sample[:, :, 0], recon_sample_random, 30, 1)

        spectrum_pred_random.append(rep_pred_random)
        spectrum_pred.append(rep_pred)
        spectrum_truth.append(rep_target)

        i += 1

    overall_truth = np.nanmean(np.array(spectrum_truth), axis=0)
    overall_pred = np.nanmean(np.array(spectrum_pred), axis=0)
    overall_pred_random = np.nanmean(np.array(spectrum_pred_random), axis=0)

    print("Average Hellinger:", np.mean(hds))
    print("Average Hellinger Random:", np.mean(hds_random))
    print("Average MSE:", np.mean(mses))
    print("Average MSE Random:", np.mean(mses_random))

    plt.plot(1/targ_freqs, overall_truth, label="Original")
    plt.plot(1/targ_freqs, overall_pred, label="Our Reconstruction")
    plt.plot(1/targ_freqs, overall_pred_random, label="Random Reconstruction")
    plt.legend()
    plt.xlabel("CRM Spacing")
    plt.ylabel(r'$\frac{m^2*crm}{s^2}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Overall signal")
    plt.savefig('./model_graphs/spectral/overall_fft_{}.pdf'.format(id))
    plt.close()

def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/config_{}.json".format(args.id))
    model_config = json.load(f)
    f.close()

    train_data = np.load(model_config["data"]["training_data_path"])
    test_data = np.load(model_config["data"]["test_data_path"])

    dataset_max = np.load(model_config["data"]["max_scalar"])
    dataset_min = np.load(model_config["data"]["min_scalar"])

    print("dataset max", dataset_max)
    print("dataset min", dataset_min)

    img_width = train_data.shape[1]
    img_height = train_data.shape[2]

    print("Image shape:", img_width, img_height)

    # Construct VAE Encoder 
    encoder_result = encoder_gen((img_width, img_height), model_config["encoder"])
    encoder_result_random = encoder_gen((img_width, img_height), model_config["encoder"])

    # Construct VAE Decoder 
    vae_decoder = decoder_gen(
        (img_width, img_height),  
        model_config["decoder"],
        encoder_result.shape_before_flattening
    )
    vae_decoder_random = decoder_gen(
        (img_width, img_height),  
        model_config["decoder"],
        encoder_result.shape_before_flattening
    )

    _, _, z = encoder_result.vae_encoder(encoder_result.inputs)
    _, _, z_random = encoder_result_random.vae_encoder(encoder_result_random.inputs)

    x_mu_var = vae_decoder(z)
    x_mu_var_random = vae_decoder_random(z_random)

    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_mu_var])
    random_vae = keras.Model(inputs=[encoder_result_random.inputs], outputs=[x_mu_var_random])

    # load weights from file
    vae.load_weights('./models/model_{}.th'.format(args.id))
    print("weights loaded")

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    compute_metrics(vae, random_vae, train_data, test_data, args.id, dataset_max, dataset_min)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the id of the config file to use to train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()