# coding: utf-8
import numpy as np


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
    #stop_condition = False
    decoded_sentence = np.zeros((max_seq_length, num_decoder_tokens))

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
        #sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #sampled_token = id2token[sampled_token_index]
        decoded_sentence[t,:] = output_tokens.copy()

        # exit condition: either hit max length
        # or find stop character.
        #if sampled_token == "<end>" or len(decoded_sentence) > max_seq_length:
        #    stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = output_tokens

    return decoded_sentence

def continue_sequence(x_start, states_value, h0, sampling, decoder_adapter_model, rnn_decoder_model, num_decoder_tokens, token2id, id2token, max_seq_length):
    len_x = int(np.sum(x_start))
    
    if h0:
        output_tokens, h, c = decoder_adapter_model.predict([x_start[0,:].reshape(1, 1, num_decoder_tokens), states_value])
        start = 1
    else:
        h = np.zeros((1,353))
        c = np.zeros((1,353))
        start = 0
    
    for t in range(start, len_x - 1):
        output_tokens, h, c = rnn_decoder_model.predict([x_start[t,:].reshape(1, 1, num_decoder_tokens), h, c])
        
    target_seq = x_start[len_x - 1,:].reshape(1, 1, num_decoder_tokens)
    stop_condition = False
    decoded_sentence = ""
    
    while not stop_condition:
        output_tokens, h, c = rnn_decoder_model.predict([target_seq, h, c])
        
        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = id2token[sampled_token_index]
        if sampling:
            sampled_token_index = np.random.choice(num_decoder_tokens, 1, p = np.squeeze(output_tokens))[0]

        # exit condition: either hit max length
        # or find stop character.
        if sampled_token == "<end>" or len(decoded_sentence) > max_seq_length:
            stop_condition = True
            decoded_sentence += sampled_token + " "
        else:
            decoded_sentence += id2token[sampled_token_index] + " "

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
    
    
    return decoded_sentence
