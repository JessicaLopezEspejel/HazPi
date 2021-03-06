import pandas as pd
import numpy as np
import tensorflow as tf
import time, sys
import re
import pickle
import argparse
import os
import copy
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


def compute_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return (tf.reduce_sum(loss_) / tf.reduce_sum(mask)) / num_gpus


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
        loss = compute_loss(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_accuracy.update_state(tar_real, predictions)

    return loss


@tf.function
def distributed_train_step(inp_dis, tar_dis):
    per_replica_losses = strategy.run(train_step, args=(inp_dis, tar_dis,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)


def increment_tokens(tar, step, step_increment_tar):
    tar = tar.numpy()
    tar[:, step * step_increment_tar:] = 0
    tar = tf.convert_to_tensor(tar, dtype=tf.int32)
    return tar


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
    parser.add_argument('-num_encoders', type=int, default=4)
    parser.add_argument('-data_path', type=str, required=True)
    parser.add_argument('-checkpoint_path', type=str, required=True)
    parser.add_argument('-vocab_load_dir', type=str, required=True)
    parser.add_argument('-epoch_extra_training', type=int, default=10)
    parser.add_argument('-epoch_inter', type=int, default=8)
    parser.add_argument('-type_ft', type=int, default=1)

    opt = parser.parse_args()

    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    print('### Number of devices: {} ...'.format(num_gpus))

    oov_token = '<unk>'

    news = pd.read_excel(opt.data_path, dtype=str)
    news.drop(['id_articles'], axis=1, inplace=True)

    documents = news['articles']
    summaries = news['abstracts']
    summaries = summaries.apply(lambda x: '<go> ' + x + ' <stop>')

    print('### Loading vocab ...')
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
        print("### Creating model..."); sys.stdout.flush();
        transformer = create_model()
        print('### Defining losses and other metrics...'); sys.stdout.flush();
        learning_rate = CustomSchedule(opt.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    with strategy.scope():
        train_loss = tf.keras.metrics.Mean(name='train_loss')

    print("### Enter to restore the checkpoint..."); sys.stdout.flush();
    ckpt_restore = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager_restore = tf.train.CheckpointManager(ckpt_restore, opt.checkpoint_path, max_to_keep=5)
    if ckpt_manager_restore.latest_checkpoint:
        ckpt_restore.restore(ckpt_manager_restore.latest_checkpoint)
        print('Latest checkpoint restored!!', ' - ' * 10)

    # End-chunk training
    step_increment_tar = round(opt.decoder_max_len / opt.epoch_inter)
    step_increment_inp = round(opt.encoder_max_len / opt.epoch_inter)

    print("step_increment_tar: ", step_increment_tar)
    if opt.type_ft == 2:
        print("step_increment_inp: ", step_increment_inp)

    with strategy.scope():
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    print("### Training..."); sys.stdout.flush();
    with strategy.scope():

        for epoch in range(opt.epoch_extra_training):

            total_loss_global = 0
            num_batches_global = 0

            for step in range(1, opt.epoch_inter + 1):

                total_loss = 0
                num_batches = 0

                for (batch, (inp, tar)) in enumerate(dataset_train):

                    tar_np = tar
                    tar_np = tar_np.numpy()

                    tar_np[:, step * step_increment_tar:] = 0
                    tar_ = tf.convert_to_tensor(tar_np, dtype=tf.int32)

                    num_batches += 1

                    if opt.type_ft == 1:
                        total_loss += distributed_train_step(inp, tar_)

                    if opt.type_ft == 2:
                        inp_np = inp
                        inp_np = inp_np.numpy()

                        inp_np[:, step * step_increment_inp:] = 0
                        inp_ = tf.convert_to_tensor(inp_np, dtype=tf.int32)

                        total_loss += distributed_train_step(inp_, tar_)

                    total_loss_global += total_loss
                    num_batches_global += 1

                    if batch % 100 == 0:
                        template = "Epoch {} Step {} Batch {}"
                        print(template.format(epoch + 1, step, batch + 1))
                        pass

                train_loss = total_loss / num_batches
                template = "Epoch {} , Step {} Batch {}, Loss: {}, Accuracy: {}"
                print(template.format(epoch + 1, step, batch + 1, train_loss, train_accuracy.result() * 100))

            train_loss_global = total_loss_global / num_batches_global
            template = "Epoch {} Loss: {} Accuracy: {}"
            print('= ' * 15, template.format(epoch + 1, train_loss_global, train_accuracy.result() * 100))

            print('### Save the checkpoints ...')
            if epoch % 1 == 0:

                path_save_ckp = opt.checkpoint_path + 'epoch_' + str(epoch + 1) + '_FT_' + str(opt.type_ft)

                if not os.path.isdir(path_save_ckp):
                    os.makedirs(path_save_ckp)
                    ckpt_manager_restore = tf.train.CheckpointManager(ckpt_restore, path_save_ckp, max_to_keep=5)
                    ckpt_save_path = ckpt_manager_restore.save()
                    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path));
                    sys.stdout.flush();