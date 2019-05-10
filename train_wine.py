# coding: utf-8

import numpy as np
from nltk.tokenize import word_tokenize
from lstm_vae import create_lstm_vae, inference
from train import get_text_data
import keras
import sys, time
from keras.callbacks import CSVLogger

from tensorflow import set_random_seed
set_random_seed(1234)
np.random.seed(1234)

from nltk.corpus import webtext
import re



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
    
    print_default = int(params['print_default'])
    
   
    timesteps_max, enc_tokens, characters, char2id, id2char, x, x_decoder = get_text_data(num_samples=num_samples,
                                                                                          data_path=data_path)

    
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
                                             latent_dim=latent_dim)
        
        csv_logger = CSVLogger('training_vae.log', separator=',', append=False)
        vae.fit([x, x_decoder], x, epochs=epochs, verbose=1, callbacks=[csv_logger])
        
        if save:
            print("Saving model ... ")
            
            vae.save("models/vae_{}_{}.h5".format(dataname, num_samples))
            enc.save("models/encoder_{}_{}.h5".format(dataname, num_samples))
            gen.save("models/generator_{}_{}.h5".format(dataname, num_samples))
            stepper.save("models/stepper_{}_{}.h5".format(dataname, num_samples))

    def decode(s, start_char = "\t"):
        return inference.decode_sequence(s, gen, stepper, input_dim, char2id, id2char, timesteps_max, start_char = start_char)

    def continue_seq(x_start, states_value, h0 = False, sampling = False):
        return inference.continue_sequence(x_start, states_value, h0, sampling, gen, stepper, input_dim, char2id, id2char, timesteps_max)
    
    if print_default:

        for _ in range(6):
            id_from = np.random.randint(0, x.shape[0] - 1)
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
            
            
raw_text = webtext.raw(webtext.fileids()[5])
listan = [sentence.replace('\t','').replace('*','').replace('\n','').strip() for sentence in re.split("[.!?]+", raw_text) if len(sentence.split())>3]


params = {}
params['batch_size'] = 1
params['latent_dim'] = 191
params['intermediate_dim'] = 353
params['epochs'] = 60
params['verbose'] = 1
params['num_samples'] = len(listan)
params['data_type'] = "text"
params['dataset'] = webtext.fileids()[5] #'wine.txt'
params['train'] = 1
params['save'] = 1
params['load'] = 0
params['print_default'] = 1

main(params)