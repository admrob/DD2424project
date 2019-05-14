# coding: utf-8

import numpy as np
from nltk.tokenize import word_tokenize
from lstm_vae import create_lstm_vae, inference
import keras
import sys, time
from keras.callbacks import CSVLogger

from tensorflow import set_random_seed
set_random_seed(1234)
np.random.seed(1234)

def get_mnist_data(num_samples=1000):
            
    # load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    n_inputs, height, max_length = X_train.shape
    encoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
    decoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
    encoder_input_data[:,:max_length,:] = np.swapaxes(X_train, 1, 2).copy()/255
    decoder_input_data[:,1:,:] = np.swapaxes(X_train, 1, 2).copy()/255

    return max_length + 1, height, encoder_input_data[:num_samples,:,:], decoder_input_data[:num_samples,:,:]

timesteps_max, enc_tokens, x, x_decoder = get_mnist_data(num_samples=60000)

print(x.shape, "Creating model...")

input_dim = x.shape[-1]
timesteps = x.shape[-2]
batch_size = 1
latent_dim = 191
intermediate_dim = 353
epochs = 40

vae, enc, gen, stepper = create_lstm_vae(input_dim,
                                         batch_size=batch_size,
                                         intermediate_dim=intermediate_dim,
                                         latent_dim=latent_dim)
print("Training model...")

vae.fit([x, x_decoder], x, epochs=epochs, verbose=1)

print("Saving model ... ")

vae.save("models/vae_mnist_60000_epoch_100.h5")
enc.save("models/encoder_mnist_epoch_100.h5")
gen.save("models/generator_mnist_epoch_100.h5")
stepper.save("models/stepper_mnist_epoch_100.h5")

print("Fitted, predicting...")


def decode(s):
    return inference.decode_sequence(s, gen, stepper, input_dim, timesteps_max)


for _ in range(5):

    id_from = np.random.randint(0, x.shape[0] - 1)
    id_to = np.random.randint(0, x.shape[0] - 1)

    m_from, std_from = enc.predict([[x[id_from]]])
    m_to, std_to = enc.predict([[x[id_to]]])

    seq_from = np.random.normal(size=(latent_dim,))
    seq_from = m_from + std_from * seq_from

    seq_to = np.random.normal(size=(latent_dim,))
    seq_to = m_to + std_to * seq_to

    print("== from \t ==")
    plt.imshow(x[id_from].T, cmap='Greys',  interpolation='nearest')
    plt.show()

    for v in np.linspace(0, 1, 7):
        print("%.2f\t" % (1 - v))
        plt.imshow(decode(v * seq_to + (1 - v) * seq_from).T, cmap='Greys',  interpolation='nearest')
        plt.show()

    print("== from \t ==")
    plt.imshow(x[id_to].T, cmap='Greys',  interpolation='nearest')
    plt.show()