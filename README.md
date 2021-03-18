# HazPi


## About this code

This repository contains the code related to the article "Saucissonnage of Long Sequences into a Multi-encoder with Transformers".
To cite this work, please use the following reference:

```
J. Lopez Espejel, G. de Chalendar, T. Charnois, J. Garcia Flores, I. Vladimir Meza Ruiz (2021),
Saucissonnage of Long Sequences into a Multi-encoder for Neural Text Summarization with Transformers,
Extraction et Gestion des Connaissances (EGC), Montpellier, France, 25-29 jan
```

## Data


We used PubMed dataset proposed in [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/pdf/1804.05685.pdf).

## How to run the baseline (![equation](https://latex.codecogs.com/svg.latex?Transformer_{ORIGINAL}))



#### Training

```bash
python ttrain_transformer.py -data_path /path_data/file.xlsx -checkpoint_path /home/path_save_checkpoints/  -vocab_save_dir /home/path_dir_vocab/ -batch_size 32 -epochs 300 -no_filters

```


Note that many codes in this repository share the same parameters, such as **num_layers**, 
**d_model**, and **dff** whose default values (precised in the code files) should not be modified in order to reproduce our results.

We explain the other parameters as follows:

1. train_transformer.py

    This code helps you to run the training of the baseline model for the first time.

    * **--checkpoint_path**. The path to save the model checkpoints
    * **--vocab_save_dir**. The path to save the vocabulary (useful to continue training from a checkpoint, or to run the evaluation with Beam Search)
    
 1. train_transformer_more_epochs.py
    
    This code helps you to load the vocabulary and a checkpoint to continue training the baseline model  

    * **-ckp_restore_path**. The checkpoint path from which the system will continue its training
    * **-ckp_save_path**. The path to save the model checkpoints
    * **-restore_epoch**. The epoch number from which the system will continue its training
    * **-vocab_load_dir**. The path of the vocabulary to load
    
1. extra_train_transformer.py

    This code helps you to run the End-Chunk Task Training (ECTT) on top of a trained baseline model.
    * **-checkpoint_path**. The checkpoint path from which the system will continue its training (sub-folders will be created to save the new model checkpoints)
    * **-vocab_load_dir**. The path of the vocabulary to load
    * **-epoch_extra_training**. The number of epochs in the ECTT
    * **-epoch_inter**. The number of chunks used to divide the gold summary when decoding 
    

####Beam Search

```bash
python beam_search_transformer.py -data_eval /path_data/file.xlsx -checkpoint_path /home/epoch_to_get_summaries/ -vocab_load_dir /home/path_dir_vocab/  -batch_size 32  -path_summaries_encoded /path/summaries/encoded/ -path_summaries_decoded /path/summaries/decoded/ -path_summaries_error /path/summaries/error/

```

## How to run our Multi-Encoder Transformer (![equation](https://latex.codecogs.com/svg.latex?HazPi))

####Training


```bash
python train_more_encoders.py -data_path /path_data/file.xlsx -checkpoint_path /home/path_save_checkpoints/  -vocab_save_dir /home/path_dir_vocab/ -epochs 300 no_filters

```

1. train_more_encoders.py

    Similarly to the baseline, we need to specify the parameters **--checkpoint_path** and **--vocab_save_dir** to create a folders in which we save the model checkpoints and the vocabulary.

1. train_more_encoders_more_epochs.py

    Similarly to the baseline, you need to specify the following parameters:
    * **-vocab_load_dir**. The path of the vocabulary to load
    * **-ckp_restore_path**. The checkpoint path from which the system will continue its training
    * **-restore_epoch**. The epoch number from which the system will continue its training
    * **-ckp_save_path**. The path we will save the following epochs 
    
1. extra_train_more_encoders.py
    
    * **-checkpoint_path** The checkpoint path from which the system will continue its training (sub-folders will be created to save the new model checkpoints)
    * **-vocab_load_dir** The path of the vocabulary to load
    * **-epoch_extra_training** The number of epochs in the ECTT
    * **-epoch_inter** The number of chunks used to divide the gold summary when decoding 

####Beam Search

```bash
python beam_search_more_encoders.py -data_eval /path_data/file.xlsx -checkpoint_path /home/epoch_to_get_summaries/ -vocab_load_dir /home/path_dir_vocab/ -path_summaries_encoded /path/summaries/encoded/ -path_summaries_decoded /path/summaries/decoded/ -path_summaries_error /path/summaries/error/

```

## Toy data example

* sample_data. 

    * medical_articles_five.xlsx file.  a toy example to train the system.
    * test_2499_3331.xlsx file. a toy example to get summaries with Beam Search.

## Remark

The code of our baseline is partially based on that in: https://github.com/rojagtap/abstractive_summarizer/
