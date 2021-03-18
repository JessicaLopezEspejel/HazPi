import pandas as pd
import numpy as np
import tensorflow as tf
import time, sys
import re
import pickle
import argparse
import os
from model import ManyEncodersTransformer
from utils import *
from scheduler import CustomSchedule

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model():
    transformer = ManyEncodersTransformer(
        opt.num_encoders,
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


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask) / num_gpus


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp,
            True,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_accuracy.update_state(tar_real, predictions)

    return loss


@tf.function
def distributed_train_step(inp_dis, tar_dis):
    per_replica_losses = strategy.run(train_step, args=(inp_dis, tar_dis,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder_max_len', type=int, default=2000)
    parser.add_argument('-decoder_max_len', type=int, default=216)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_layers', type=int, default=4)
    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-dff', type=int, default=2048)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-encoder_max_vocab', type=int, default=100000)
    parser.add_argument('-decoder_max_vocab', type=int, default=100000)
    parser.add_argument('-data_path', type=str, required=True)
    parser.add_argument('-num_encoders', type=int, default=4)
    parser.add_argument('-vocab_load_dir', type=str, required=True)
    parser.add_argument('-ckp_restore_path', type=str, required=True)
    parser.add_argument('-ckp_save_path', type=str, required=True)
    parser.add_argument('-restore_epoch', type=int, required=True)

    opt = parser.parse_args()

    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    print('### Number of devices: {} ... '.format(num_gpus)); sys.stdout.flush();

    oov_token = '<unk>'

    print('### Loading data ...')
    news = pd.read_excel(opt.data_path, dtype=str)
    news.drop(['id_articles'], axis=1, inplace=True)
    documents = news['articles']
    summaries = news['abstracts']
    summaries = summaries.apply(lambda x: '<go> ' + x + ' <stop>')

    print('Loading vocab')
    with open(os.path.join(opt.vocab_load_dir) + 'document_tokenizer_{}.pickle'.format(opt.encoder_max_vocab),
              'rb') as fp:
        document_tokenizer = pickle.load(fp)

    with open(os.path.join(opt.vocab_load_dir) + 'summary_tokenizer_{}.pickle'.format(opt.decoder_max_vocab),
              'rb') as fp:
        summary_tokenizer = pickle.load(fp)

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

    print("### Obtaining insights on lengths for defining maxlen..."); sys.stdout.flush();
    document_lengths = pd.Series([len(x) for x in documents])
    summary_lengths = pd.Series([len(x) for x in summaries])
    BUFFER_SIZE = int(document_lengths.count())

    print("### Padding/Truncating sequences for identical sequence lengths..."); sys.stdout.flush();
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=opt.encoder_max_len, padding='post',
                                                           truncating='post')
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=opt.decoder_max_len, padding='post',
                                                            truncating='post')

    print("### Creating dataset pipeline..."); sys.stdout.flush();
    inputs = tf.cast(inputs, dtype=tf.int32)
    targets = tf.cast(targets, dtype=tf.int32)

    dataset_train = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(opt.batch_size)
    # train_dist_dataset = strategy.experimental_distribute_dataset(dataset_train)

    with strategy.scope():
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                    reduction=tf.keras.losses.Reduction.NONE)

    with strategy.scope():
        print("### Creating model..."); sys.stdout.flush();
        transformer = create_model()
        print('### Defining losses and other metrics...'); sys.stdout.flush();
        learning_rate = CustomSchedule(opt.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    ckpt_restore = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager_restore = tf.train.CheckpointManager(ckpt_restore, opt.ckp_restore_path, max_to_keep=5)
    if ckpt_manager_restore.latest_checkpoint:
        ckpt_restore.restore(ckpt_manager_restore.latest_checkpoint)
        print('### Latest checkpoint restored!! ...')

    with strategy.scope():
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    print("### Training..."); sys.stdout.flush();
    with strategy.scope():

        for epoch in range(opt.epochs):

            total_loss = 0.0
            num_batches = 0
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(dataset_train):

                total_loss += distributed_train_step(inp, tar)
                num_batches += 1

                if num_batches % 1000 == 0:
                    print('Epoch {} Batch {}'.format(epoch + 1, batch + 1)); sys.stdout.flush();

            train_loss = total_loss / num_batches

            template = "Epoch {}, Loss: {}, Accuracy: {}"
            print(template.format(epoch + 1 + opt.restore_epoch, train_loss, train_accuracy.result() * 100));
            sys.stdout.flush();

            print('### Save the checkpoints ...')
            if (epoch + 1 + opt.restore_epoch) % 1 == 0:
                path_save_ckp = opt.ckp_save_path + 'epoch_' + str(epoch + 1 + opt.restore_epoch)

                if not os.path.isdir(path_save_ckp):
                    os.makedirs(path_save_ckp)
                    ckpt_manager = tf.train.CheckpointManager(ckpt_restore, path_save_ckp, max_to_keep=5)
                    ckpt_save_path = ckpt_manager.save()
                    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1 + opt.restore_epoch, ckpt_save_path));
                    sys.stdout.flush();
