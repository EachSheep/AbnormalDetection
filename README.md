# Abnormal Detection

## Setup 

Install with `pip install -r requirements.txt`.

Or follow instructions in requirements.txt to install.

## Running

### train 

训练的时候，主要可调的参数有**max_seq_len, embedding_dim, hidden_dim, steps_per_epoch, batch_size**

```bash
# 第一次运行时生成cache文件
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                -max_seq_len 200 \
                -vocab_dict_path data/assets/page2idx.json \
                -vocab_size 10000 \
                -backbone lstma \
                -embedding_dim 280 \
                -hidden_dim 200 \
                -criterion BCE \
                -lr 0.0002 \
                -epochs 30 \
                -steps_per_epoch 40 \
                -batch_size 128 \
                -train_ratio 0.8

# 之后的运行可以制定cache参数
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                --use_cache \
                -max_seq_len 200 \
                -vocab_dict_path data/assets/page2idx.json \
                -vocab_size 10000 \
                -backbone lstma \
                -embedding_dim 280 \
                -hidden_dim 200 \
                -criterion BCE \
                -lr 0.0002 \
                -epochs 30 \
                -steps_per_epoch 40 \
                -batch_size 128 \
                -train_ratio 0.8 \
                > experiment/log.txt

# 更换backbone为lstm
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                --use_cache \
                -max_seq_len 200 \
                -vocab_dict_path data/assets/page2idx.json \
                -vocab_size 10000 \
                -backbone lstm \
                -n_layers 3 \
                -embedding_dim 280 \
                -hidden_dim 200 \
                -criterion BCE \
                -lr 0.0002 \
                -epochs 30 \
                -steps_per_epoch 40 \
                -batch_size 128 \
                -train_ratio 0.8

# BCE, focal, deviation
```

transformer

```bash

python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                --use_cache \
                -max_seq_len 200 \
                -vocab_dict_path data/assets/page2idx.json \
                -vocab_size 10000 \
                -backbone transformer \
                -key_size 200 \
                -query_size 200 \
                -value_size 200 \
                -num_hiddens 280 \
                -norm_shape 2 \
                -ffn_num_input 200 \
                -ffn_num_hiddens 200 \
                -num_heads 4 \
                -num_layers 2 \
                -dropout 0.5 \
                -criterion BCE \
                -lr 0.0002 \
                -epochs 30 \
                -steps_per_epoch 40 \
                -batch_size 128 \
                -train_ratio 0.8
```

### test

```bash
python test.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/ \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                -max_seq_len 200 \
                -vocab_dict_path pre/data/page2idx.json > experiment/log.txt
```

## Reference

["Explainable Deep Few-shot Anomaly Detection with Deviation Networks"](https://arxiv.org/abs/2108.00462)