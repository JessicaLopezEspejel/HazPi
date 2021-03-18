import pandas as pd
import numpy as np
import tensorflow as tf
import time
import re
import pickle
import os
import sys
from collections import Counter
import math
import collections
from collections import defaultdict

from model import Transformer
from utils import *
from scheduler import CustomSchedule
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def check_repeated_ngrams(text, n):
    text = text.split()
    ngrams = []
    i = 0
    while i < len(text):
        if len(text[i:i+n]) >= n :
            ngrams.append(' '.join(text[i:i+n]))
        i += 1
    repeated_ngrams = [item for item, count in collections.Counter(ngrams).items() if count > 1]
    return repeated_ngrams


def create_model():
    transformer = Transformer(
        opt.num_layers,
        opt.d_model,
        opt.num_heads,
        opt.dff,
        encoder_vocab_size,
        decoder_vocab_size,
        pe_input=encoder_vocab_size,
        pe_target=decoder_vocab_size,
    )
    return transformer


def read_articles_summarize(input_document, document_tokenizer, opt):
    with tf.device('/GPU:0'):
        input_document = document_tokenizer.texts_to_sequences([input_document])
        input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=opt.encoder_max_len,
                                                                       padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], axis=0)
    return encoder_input


def initialization_vars(encoder_input, summary_tokenizer, transformer, opt):
    decoder_input = [summary_tokenizer.word_index["<go>"]]
    outputs = tf.expand_dims(decoder_input, 0)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, outputs)
    predictions, attention_weights = transformer(encoder_input, outputs, False, enc_padding_mask,
                                                 combined_mask, dec_padding_mask)
    predictions = predictions[:, -1:, :]
    predictions = tf.keras.activations.softmax(predictions, axis=-1)
    with tf.device('/GPU:0'):
        probs, ind = tf.math.top_k(predictions, opt.k, sorted=True)
    log_scores = tf.cast([math.log(prob) for prob in probs.numpy().flatten()], dtype=tf.float32)
    log_scores = tf.expand_dims(log_scores, axis=0)
    outputs = np.zeros((opt.k, opt.len_summary))

    outputs[:, 0] = decoder_input
    outputs[:, 1] = ind[0]
    outputs = tf.cast(outputs, dtype=tf.int32)

    return outputs, log_scores


def k_best_ouputs(outputs, probs_matrix_np, ind_matrix_np, log_scores, i, opt):

    log_probs = np.array([math.log(prob) for prob in probs_matrix_np.flatten()]).reshape(opt.k, opt.k) + \
                 tf.keras.backend.transpose(log_scores).numpy()
    with tf.device('/GPU:0'):
        log_probs = tf.expand_dims(tf.cast(log_probs.flatten(), dtype=tf.float32), axis=0)
    with tf.device('/GPU:0'):
        probs, ind = tf.math.top_k(log_probs, opt.k, sorted=True)

    row = ind // opt.k
    col = ind % opt.k

    outputs = outputs.numpy()

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ind_matrix_np[row, col]
    outputs = tf.cast(outputs, dtype=tf.int32)
    log_scores = probs

    return outputs, log_scores


def summarize(input_document, summary_tokenizer, document_tokenizer, transformer, opt):
    stop_token = [summary_tokenizer.word_index["<stop>"]]
    stop_token = tf.expand_dims(stop_token, 0)

    with tf.device('/GPU:0'):
        encoder_input = read_articles_summarize(input_document, document_tokenizer, opt)
        outputs, log_scores = initialization_vars(encoder_input, summary_tokenizer, transformer, opt)
    ind = None

    # Initialization of dic_ngrams
    dic_ngrams = defaultdict(list)
    for i in range(opt.k):
        dic_ngrams[i] = collections.Counter()

    for i in range(2, opt.len_summary):

        ind_matrix_np = np.zeros((opt.k, opt.k))
        probs_matrix_np = np.zeros((opt.k, opt.k))

        for j in range(opt.k):

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, outputs[j:j+1, :i])

            predictions, attention_weights = transformer(encoder_input, outputs[j:j+1,:i], False, enc_padding_mask,
                                                         combined_mask, dec_padding_mask)

            predictions = predictions[:, -1:, :]
            predictions = tf.keras.activations.softmax(predictions, axis=-1)

            if opt.ngram_size > 1:
                current_seq = outputs[j:j + 1, :i].numpy()

                if i >= opt.ngram_size + 1:
                    # Get all the ngrams
                    for k in range(i - opt.ngram_size + 1):
                        ngram = tuple(current_seq[0, k: k + opt.ngram_size])
                        dic_ngrams[j][ngram] += 1

                    predictions = predictions.numpy()
                    index_last_token = int(current_seq[0, i - 1])
                    predictions[:, :, index_last_token] = 0
                    ngram_part = current_seq[0, i - opt.ngram_size + 1: i].tolist()
                    arg_sorted_predictions = tf.argsort(predictions, axis=-1, direction='DESCENDING').numpy().tolist()[0][0]

                    valid_indices = 0
                    for index_token in arg_sorted_predictions:
                        if dic_ngrams[j][tuple(ngram_part + [index_token])] > 0:
                            predictions[:, :, index_token] = 0
                        else:
                            valid_indices += 1
                            if valid_indices == opt.k:
                                break

                    predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)

            probs, ind_ = tf.math.top_k(predictions, opt.k, sorted=True)
            probs_matrix_np[j,:] = probs
            ind_matrix_np[j, :] = ind_

        outputs, log_scores = k_best_ouputs(outputs, probs_matrix_np, ind_matrix_np, log_scores, i, opt)

        ones = tf.where((outputs == stop_token))
        sentence_lengths_np = np.zeros((len(outputs)))

        for vec in ones:
            i = vec[0]
            if sentence_lengths_np[i] == 0:
                sentence_lengths_np[i] = vec[1]

        num_finished_sentences = len([s for s in sentence_lengths_np if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.8
            div = 1 / (tf.cast(sentence_lengths_np, dtype=tf.float32) ** alpha)
            ind = tf.keras.backend.argmax(log_scores * div, axis=1)

    if ind is None:
        print('Not all sentences finished with End-of-Seq symbol')
        ones = tf.where((outputs == stop_token))
        try:
            ind = ones[0][0]
            length = tf.where((outputs[ind] == stop_token))[0][1]
            print('At least one solution finished with End-of-Seq symbol')
        except TypeError:
            length = tf.where((outputs[ind] == stop_token)).numpy()[1]
            print('At least one solution finished with End-of-Seq symbol')
        except:
            print('No sentence finished with End-of-Seq symbol')  # the two first cases of the articles are HERE
            ind = 0
            length = len(outputs[ind])
    else:
        try:
            length = tf.where((outputs[ind] == stop_token))[1]
        except:
            ind = 0
            length = tf.where((outputs[ind] == stop_token))[0][1]

    summary = outputs[ind][1:length].numpy().tolist()

    return summary


def summarize_text(outputs, summary_tokenizer):
    summary = summary_tokenizer.sequences_to_texts(outputs)
    return summary


def write_summary(path, summary_txt, num_summary, tag):
    path_txt = path + str(num_summary).zfill(6) + '_' + tag + '.txt'
    with open(path_txt, 'w') as writer:
        writer.write(summary_txt)


def write_error(path, doc, gold_summary, error, num_summary):
    doc_path = os.path.join(path, str(num_summary).zfill(6) + '_doc.txt')
    gold_path = os.path.join(path, str(num_summary).zfill(6) + '_gold.txt')

    try:
        with open(doc_path, 'w') as doc_writer:
            doc_writer.write(doc)

        with open(gold_path, 'w') as gold_writer:
            gold_writer.write(gold_summary)

        print('Could not summarize document number {}'.format(num_summary))
        print('Error = {}'.format(error))
        print('Doc saved in {}'.format(doc_path))
        print('Gold saved in {}'.format(gold_path))
        print('-----------------------------')
        sys.stdout.flush()

    except Exception as err:
        print('Could not print document number {}'.format(num_summary))
        print('Error = {}'.format(err))
        print('-----------------------------')
        sys.stdout.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder_max_len', type=int, default=2000)
    parser.add_argument('-decoder_max_len', type=int, default=216)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_layers', type=int, default=4)
    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-dff', type=int, default=2048)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-encoder_max_vocab', type=int, default=100000)
    parser.add_argument('-decoder_max_vocab', type=int, default=100000)
    parser.add_argument('-data_eval', type=str, required=True)
    parser.add_argument('-vocab_load_dir', type=str, required=True)
    parser.add_argument('-checkpoint_path', type=str, required=True)
    parser.add_argument('-path_summaries_encoded', type=str, required=True)
    parser.add_argument('-path_summaries_decoded', type=str, required=True)
    parser.add_argument('-path_summaries_error', type=str, required=True)
    parser.add_argument('-len_summary', type=int, default=216)
    parser.add_argument('-k', type=int, default=6)
    parser.add_argument('-ngram_size', type=int, default=2)

    opt = parser.parse_args()

    assert(os.path.exists(opt.vocab_load_dir))

    oov_token = '<unk>'
    num_articles = 1

    docs_eval = pd.read_excel(opt.data_eval, dtype=str)
    docs_eval.drop(['id_articles'], axis=1, inplace=True)
    docs_eval.articles = docs_eval.articles.astype(str)
    docs_eval.abstracts = docs_eval.abstracts.astype(str)

    documents = docs_eval['articles']
    summaries = docs_eval['abstracts']

    print('### Loading vocab ...')
    with open(os.path.join(opt.vocab_load_dir) + 'document_tokenizer_{}.pickle'.format(opt.encoder_max_vocab), 'rb') as fp:
        document_tokenizer = pickle.load(fp)

    with open(os.path.join(opt.vocab_load_dir) + 'summary_tokenizer_{}.pickle'.format(opt.decoder_max_vocab), 'rb') as fp:
        summary_tokenizer = pickle.load(fp)

    with tf.device('/GPU:0'):
        inputs = document_tokenizer.texts_to_sequences(documents)
        targets = summary_tokenizer.texts_to_sequences(summaries)

    if opt.encoder_max_vocab != -1:
        encoder_vocab_size = opt.encoder_max_vocab
    else:
        encoder_vocab_size = len(document_tokenizer.word_index) + 1

    if opt.decoder_max_vocab != -1:
        decoder_vocab_size = opt.decoder_max_vocab
    else:
        decoder_vocab_size = len(summary_tokenizer.word_index) + 1

    print("### Obtaining insights on lengths for defining maxlen..."); sys.stdout.flush()
    document_lengths = pd.Series([len(x) for x in documents])
    summary_lengths = pd.Series([len(x) for x in summaries])
    BUFFER_SIZE = int(document_lengths.count())

    print("### Padding/Truncating sequences for identical sequence lengths..."); sys.stdout.flush()
    with tf.device('/GPU:0'):
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=opt.encoder_max_len, padding='post',
                                                               truncating='post')
        targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=opt.decoder_max_len, padding='post',
                                                                truncating='post')

    # To know how many <unk> we have in the input and target sequences.
    cnt_unk_inputs = cnt_unk_targets = 0
    for lst_input, lst_target in zip(inputs.tolist(), targets.tolist()):
        cnt_unk_inputs += Counter(lst_input)[1]
        cnt_unk_targets += Counter(lst_target)[1]

    print("### Creating dataset pipeline ..."); sys.stdout.flush()
    inputs = tf.cast(inputs, dtype=tf.int32)
    targets = tf.cast(targets, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(opt.batch_size)

    print('### Defining losses and other metrics...'); sys.stdout.flush()
    learning_rate = CustomSchedule(opt.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    print("### Creating model..."); sys.stdout.flush()
    with tf.device('/GPU:0'):
        transformer = create_model()

    print("### Setting checkpoints manager..."); sys.stdout.flush()
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, opt.checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
       ckpt.restore(ckpt_manager.latest_checkpoint)
       print('Latest checkpoint restored!!'); sys.stdout.flush()

    print("### Evaluation..."); sys.stdout.flush()
    documents_pass = list(documents)
    summary_gold = list(docs_eval['abstracts'])

    if not os.path.exists(opt.path_summaries_decoded):
        os.makedirs(opt.path_summaries_decoded)

    if not os.path.exists(opt.path_summaries_encoded):
        os.makedirs(opt.path_summaries_encoded)

    if not os.path.exists(opt.path_summaries_error):
        os.makedirs(opt.path_summaries_error)

    ind_sum = int(opt.data_eval.split('/')[-1].split('.')[0].split('_')[1])

    for idx, (doc, gold_summary) in enumerate(zip(documents_pass, summary_gold)):
        try:
            start_time = time.time()
            summary = summarize(doc, summary_tokenizer, document_tokenizer, transformer, opt)
            summary_generated = summarize_text([summary], summary_tokenizer)[0]
            if opt.ngram_size != -1:
                repeated_ngrams = check_repeated_ngrams(summary_generated, opt.ngram_size)
                print(repeated_ngrams)
                print()
            print("SUMMARY TIME: ", time.time() - start_time)

            write_summary(opt.path_summaries_encoded, gold_summary, ind_sum + idx, 'encoder')
            write_summary(opt.path_summaries_decoded, summary_generated, ind_sum + idx, 'decoder')

        except Exception as error:
            write_error(opt.path_summaries_error, doc, gold_summary, error, ind_sum + idx)

    print("Summarization Finished!")
