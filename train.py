# coding: utf-8

import numpy as np
from nltk.tokenize import word_tokenize
from lstm_vae import create_lstm_vae, inference
import keras
import matplotlib.pyplot as plt

dataset = 'iamondb'

def load_sequences(source_dir, seq_file='sequences.npy', idx_file='sequence_indices.npy'):
    """ load data mat and length max, add lengths up to get indices """
    seq_mat = np.load(source_dir + '/' + seq_file)
    idx_mat = np.load(source_dir + '/' + idx_file)
    # plt.hist(idx_mat, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.show()
    for idx in range(1, idx_mat.shape[0]):
        idx_mat[idx] = idx_mat[idx] + idx_mat[idx - 1]

    return seq_mat, idx_mat

def load_and_cut_sequences(source_dir, seq_file='train_sequences.npy', idx_file='train_sequence_indices.npy', cut_len=500,
                           normalize=True, mask=True, mask_value=500):
    """ loads sequences, cuts to certain lenght. optionally normalizing and masking them """
    if not mask:
        mask_value = 0

    seq_mat, idx_mat = load_sequences(source_dir, seq_file, idx_file)

    split_list = np.split(seq_mat.astype(float), idx_mat[1:], axis=0)

    # cut sequences to maximum length
    cut_list = []
    for mat in split_list:
        if mat.shape[0] > cut_len:
            cut_list.append(mat[:cut_len, :])
        else:
            cut_list.append(mat)

    cut_seq_mat = np.concatenate(cut_list, axis=0)
    
    # compute adequate mean and std-dev
    if normalize:
        if mask:
            cut_mat = cut_seq_mat.astype(float)
        else:  # append as many zeros, as will be padded to cut_seq_mat
            zero_shape = (len(cut_list) * cut_len - cut_seq_mat.shape[0], cut_seq_mat.shape[1])
            cut_mat = np.concatenate([cut_seq_mat.astype(float), np.zeros(zero_shape, dtype=float)], axis=0)

        mean = np.mean(cut_mat, axis=0)
        std = np.std(cut_mat, axis=0)
        # for idx in [0, 1]:
        #     mat[:, idx] = (mat[:, idx] - mean[idx]) / std[idx]

    else:
        mean = [0., 0., 0.]
        std = [1., 1., 1.]

    # normalize and pad sequences
    for idx, mat in enumerate(cut_list):
        if normalize:
            mat[:, 0] = (mat[:, 0] - mean[0]) / std[0]
            mat[:, 1] = (mat[:, 1] - mean[1]) / std[1]

        if mat.shape[0] < cut_len:
            padded = np.zeros((cut_len, 3), dtype=float) + mask_value
            padded[:mat.shape[0], :] = mat
            mat = padded

        cut_list[idx] = mat

    data_mat = np.asarray(cut_list)
    
    return data_mat, mean, std


def get_dataset(num_samples=1000, data_type = 'mnist'):
            
    # load data
    if data_type == 'mnist':
        (X_train, _), _ = keras.datasets.mnist.load_data()
        n_inputs, height, max_length = X_train.shape
        encoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
        decoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
        encoder_input_data[:,:max_length,:] = np.swapaxes(X_train, 1, 2).copy()/255
        decoder_input_data[:,1:,:] = np.swapaxes(X_train, 1, 2).copy()/255
        
    elif data_type == 'iamondb':
        data_mat, mean, stp = load_and_cut_sequences(source_dir = 'data/handwriting', mask = False, cut_len = 2000, normalize = False)
        n_inputs, height, max_length = data_mat.shape
        encoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
        decoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")

    return max_length + 1, height, encoder_input_data[:num_samples,:,:], decoder_input_data[:num_samples,:,:]


if __name__ == "__main__":

    timesteps_max, enc_tokens, x, x_decoder = get_dataset(num_samples=3000, data_type = dataset)

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