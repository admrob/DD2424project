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


def get_text_data(num_samples=1000):
            
    # load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    n_inputs, height, max_length = X_train.shape
    encoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
    decoder_input_data = np.zeros((n_inputs, max_length + 1, height), dtype="float32")
    encoder_input_data[:,:max_length,:] = np.swapaxes(X_train, 1, 2).copy()/255
    decoder_input_data[:,1:,:] = np.swapaxes(X_train, 1, 2).copy()/255

    # vectorize the data
    input_texts = []
    input_characters = set(["\t"])

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().lower().split("\n")

    for line in lines[: min(num_samples, len(lines) - 1)]:
        if 'fra.txt' in data_path:

            input_text, _ = line.split("\t")
            input_text = word_tokenize(input_text)
            input_text.append("<end>")
        else:
            input_text = line.split()

        input_texts.append(input_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)

    input_characters = sorted(list(input_characters))
    num_encoder_tokens = len(input_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts]) + 1

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    decoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")

    for i, input_text in enumerate(input_texts):
        decoder_input_data[i, 0, input_token_index["\t"]] = 1.0

        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
            decoder_input_data[i, t + 1, input_token_index[char]] = 1.0

    return max_encoder_seq_length, num_encoder_tokens, input_characters, input_token_index, reverse_input_char_index, \
           encoder_input_data, decoder_input_data


def main(params):
    
    num_samples = int(params['num_samples'])
    data_path = "data/" + params['dataset']
    dataname = params['dataset'].split('.')[0]
    
    batch_size = int(params['batch_size'])
    latent_dim = int(params['latent_dim'])
    intermediate_dim = int(params['intermediate_dim'])
    epochs = int(params['epochs'])
    
    train = int(params['train'])
    save = int(params['save'])
    load = int(params['load'])

    timesteps_max, enc_tokens, characters, char2id, id2char, x, x_decoder = get_text_data(num_samples=num_samples,
                                                                                          data_path=data_path)

    print(x.shape, "Creating model...")
    
    input_dim = x.shape[-1]
    timesteps = x.shape[-2]
    
    if load:
        print("Loading model ... ")
        
        #vae = keras.models.load_model("models/vae_{}_{}.h5".format(dataname, num_samples))
        enc = keras.models.load_model("models/encoder_{}_{}.h5".format(dataname, num_samples))
        gen = keras.models.load_model("models/generator_{}_{}.h5".format(dataname, num_samples))
        stepper = keras.models.load_model("models/stepper_{}_{}.h5".format(dataname, num_samples))
    
    if train:
        print("Training model...")
        
        vae, enc, gen, stepper = create_lstm_vae(input_dim,
                                             batch_size=batch_size,
                                             intermediate_dim=intermediate_dim,
                                             latent_dim=latent_dim,
                                             data_type = data_type)
        
        csv_logger = CSVLogger('training_vae.log', separator=',', append=False)
        vae.fit([x, x_decoder], x, epochs=epochs, verbose=1, callbacks=[csv_logger])
        
        if save:
            print("Saving model ... ")
            
            vae.save("models/vae_{}_{}.h5".format(dataname, num_samples))
            enc.save("models/encoder_{}_{}.h5".format(dataname, num_samples))
            gen.save("models/generator_{}_{}.h5".format(dataname, num_samples))
            stepper.save("models/stepper_{}_{}.h5".format(dataname, num_samples))
    
    print("Fitted, predicting...")


    def decode(s, start_char = "\t"):
            return inference.decode_sequence(s, gen, stepper, input_dim, timesteps_max, char2id, id2char, start_char = start_char, data_type = data_type)

    def continue_seq(x_start, states_value, h0 = False, sampling = False):
        return inference.continue_sequence(x_start, states_value, h0, sampling, gen, stepper, input_dim, char2id, id2char, timesteps_max)

    for _ in range(5):

        id_from = np.random.randint(0, x.shape[0] - 1)
        id_to = np.random.randint(0, x.shape[0] - 1)

        m_from, std_from = enc.predict([[x[id_from]]])
        m_to, std_to = enc.predict([[x[id_to]]])

        seq_from = np.random.normal(size=(latent_dim,))
        seq_from = m_from #+ std_from * seq_from

        seq_to = np.random.normal(size=(latent_dim,))
        seq_to = m_to #+ std_to * seq_to

        if data_type == 'mnist':
            print("== from \t ==")  
            plt.imshow(x[id_from].T, cmap='Greys',  interpolation='nearest')
            plt.grid(False)
            plt.show()
    
            for v in np.linspace(0, 1, 7):
                print("%.2f\t" % (1 - v))
                plt.imshow(decode(v * seq_to + (1 - v) * seq_from).T, cmap='Greys',  interpolation='nearest')
                plt.grid(False)
                plt.show()
    
            print("== to \t ==")
            plt.imshow(x[id_to].T, cmap='Greys',  interpolation='nearest')
            plt.grid(False)
            plt.show()
        
        elif data_type == 'text':
            print("==  \t", " ".join([id2char[j] for j in np.argmax(x[id_from], axis=1)]), "==")
    
            for v in np.linspace(0, 1, 7):
                print("%.2f\t" % (1 - v), decode(v * seq_to + (1 - v) * seq_from))
    
            print("==  \t", " ".join([id2char[j] for j in np.argmax(x[id_to], axis=1)]), "==")
       
    if data_type == 'text':
        for _ in range(6):
            id_sentence = np.random.randint(0, x.shape[0] - 1)
            
            n_words = np.sum(x[id_sentence])
            n_kept = np.random.randint(n_words//2, n_words-1)
            
            new_x = np.zeros((x[id_sentence].shape))
            new_x[:n_kept,:] = x[id_sentence,:n_kept,:]
            
            m_new, std_new = enc.predict([[x[id_from]]])
            h_new = np.random.normal(size=(latent_dim,))
            states_new = m_new + std_new * h_new
            
            print("==  \t", " ".join([id2char[j] for j in np.argmax(new_x[:n_kept], axis=1)]), " ... \t\t ==")
            
            print("\t...\t", continue_seq(new_x, states_new))
            print("\t...\t", continue_seq(new_x, states_new, h0 = True))
            
            print("\t...\t", continue_seq(new_x, states_new, sampling = True))
            print("\t...\t", continue_seq(new_x, states_new, h0 = True, sampling = True))
                
                
            print("==  \t", " ".join([id2char[j] for j in np.argmax(x[id_sentence], axis=1)]), "==")
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'config.txt'

    with open(config_file_name, 'r') as f:
        lines = f.readlines()
        
    params = {}

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    main(params)
