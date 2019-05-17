# coding: utf-8

import numpy as np
from nltk.tokenize import word_tokenize
import keras
import sys, time
from keras.callbacks import CSVLogger

from tensorflow import set_random_seed
set_random_seed(1234)
np.random.seed(1234)

# coding: utf-8

from keras import backend as K
from keras import objectives
from keras.layers import Input, LSTM
from keras.layers.core import Dense, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects


def decode_sequence(states_value, decoder_adapter_model, rnn_decoder_model, num_decoder_tokens, token2id, id2token, max_seq_length, start_char = "\t"):
    """
    Decoding adapted from this example:
    https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

    :param states_value:
    :param decoder_adapter_model: reads text representation, makes the first prediction, yields states after the first RNN's step
    :param rnn_decoder_model: reads previous states and makes one RNN step
    :param num_decoder_tokens:
    :param token2id: dict mapping words to ids
    :param id2token: dict mapping ids to words
    :param max_seq_length: the maximum length of the sequence
    :return:
    """

    # generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # populate the first token of the target sequence with the start character
    target_seq[0, 0, token2id[start_char]] = 1.0

    # sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1)
    stop_condition = False
    decoded_sentence = ""

    first_time = True
    h, c = None, None
        

    while not stop_condition:
        if first_time:
            # feeding in states sampled with the mean and std provided by encoder
            # and getting current LSTM states to feed in to the decoder at the next step
            output_tokens, h, c = decoder_adapter_model.predict([target_seq, states_value])
            first_time = False
        else:
            # reading output token
            output_tokens, h, c = rnn_decoder_model.predict([target_seq, h, c])

        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = id2token[sampled_token_index]
        decoded_sentence += sampled_token + " "

        # exit condition: either hit max length
        # or find stop character.
        if sampled_token == "<end>" or len(decoded_sentence) > max_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

    return decoded_sentence

def create_lstm_vae(input_dim,
                    batch_size,  # we need it for sampling
                    intermediate_dim,
                    latent_dim):
    """
    Creates an LSTM Variational Autoencoder (VAE).

    # Arguments
        input_dim: int.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.


    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(None, input_dim,))

    # LSTM encoding
    h = LSTM(units=intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(units=latent_dim)(h)
    z_log_sigma = Dense(units=latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    z_reweighting = Dense(units=intermediate_dim, activation="linear")
    z_reweighted = z_reweighting(z)

    # "next-word" data for prediction
    decoder_words_input = Input(shape=(None, input_dim,))

    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True, return_state=True)

    # todo: not sure if this initialization is correct
    h_decoded, _, _ = decoder_h(decoder_words_input, initial_state=[z_reweighted, z_reweighted])
    decoder_dense = TimeDistributed(Dense(input_dim, activation="sigmoid"))
    decoded_onehot = decoder_dense(h_decoded)

    # end-to-end autoencoder
    vae = Model([x, decoder_words_input], decoded_onehot)

    # encoder, from inputs to latent space
    encoder = Model(x, [z_mean, z_log_sigma])

    # generator, from latent space to reconstructed inputs -- for inference's first step
    decoder_state_input = Input(shape=(latent_dim,))
    _z_rewighted = z_reweighting(decoder_state_input)
    _h_decoded, _decoded_h, _decoded_c = decoder_h(decoder_words_input, initial_state=[_z_rewighted, _z_rewighted])
    _decoded_onehot = decoder_dense(_h_decoded)
    generator = Model([decoder_words_input, decoder_state_input], [_decoded_onehot, _decoded_h, _decoded_c])

    # RNN for inference
    input_h = Input(shape=(intermediate_dim,))
    input_c = Input(shape=(intermediate_dim,))
    __h_decoded, __decoded_h, __decoded_c = decoder_h(decoder_words_input, initial_state=[input_h, input_c])
    __decoded_onehot = decoder_dense(__h_decoded)
    stepper = Model([decoder_words_input, input_h, input_c], [__decoded_onehot, __decoded_h, __decoded_c])

    def vae_loss(x, x_decoded_onehot):
        xent_loss = objectives.categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss
    
    def xent_loss(x, x_decoded_onehot):
        xent_loss = objectives.categorical_crossentropy(x, x_decoded_onehot)
        return xent_loss
    

    def kl_loss(x, x_decoded_onehot):
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        return kl_loss

    def combined_loss(x, x_decoded_onehot):
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        mse_loss = objectives.mean_squared_error(x, x_decoded_onehot)
        return mse_loss+kl_loss
    
    def mse_loss(x, x_decoded_onehot):
        mse_loss = objectives.mean_squared_error(x, x_decoded_onehot)
        return mse_loss
    
    get_custom_objects().update({"combined_loss": combined_loss,"mse_loss": mse_loss, 'kl_loss':kl_loss})

    vae.compile(optimizer="adam", loss=combined_loss, metrics = [combined_loss, mse_loss, kl_loss])
    vae.summary()

    return vae, encoder, generator, stepper


def get_mnist_data(num_samples=1000):
            
    # load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    n_inputs, height, max_length = X_train.shape
    encoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
    decoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
    encoder_input_data[:,:max_length,:] = np.swapaxes(X_train, 1, 2).copy()/255
    decoder_input_data[:,1:,:] = np.swapaxes(X_train, 1, 2).copy()/255

    return max_length + 1, height, encoder_input_data[:num_samples,:,:], decoder_input_data[:num_samples,:,:]

timesteps_max, enc_tokens, x, x_decoder = get_mnist_data(num_samples=30000)

print(x.shape, "Creating model...")

input_dim = x.shape[-1]
timesteps = x.shape[-2]
batch_size = 1
latent_dim = 1714
intermediate_dim = 976
epochs = 40

vae, enc, gen, stepper = create_lstm_vae(input_dim,
                                         batch_size=batch_size,
                                         intermediate_dim=intermediate_dim,
                                         latent_dim=latent_dim)

print("Training model...")

vae.fit([x, x_decoder], x, epochs=epochs, verbose=1)

print("Saving model ... ")

vae.save("models/vae_mnist_30000_epoch_40.h5")
enc.save("models/encoder_mnist_30000_epoch_40.h5")
gen.save("models/generator_mnist_30000_epoch_40.h5")
stepper.save("models/stepper_mnist_30000_epoch_40.h5")

print("Fitted, predicting...")


def decode(s):
    return inference.decode_sequence(s, gen, stepper, input_dim, timesteps_max)
