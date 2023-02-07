# Abnormal Detection

## Setup 

Install with `pip install -r requirements.txt`.

Or follow instructions in requirements.txt to install.

## Train

### lstm based 

训练的时候，主要可调的参数有**max_seq_len, embedding_dim, hidden_dim, steps_per_epoch, batch_size**

```bash
# 第一次运行时生成cache文件，注意！！！更改max_seq_len之后需要去掉--cache重新运行一遍
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -weight_name model.pkl \
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

# 之后的运行可以指定cache参数。
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -weight_name model.pkl \
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
                -train_ratio 0.8

# 可以使用的loss有BCE, focal, deviation，但是感觉BCE就够了
```

### transformer based 

transformer为backbone。

训练的时候，主要可调的参数有**max_seq_len, embedding_dim, ffn_num_hiddens, num_heads, num_layers, batch_size**。

注意：embedding_dim必须是num_heads的倍数


```bash
# 第一次运行时生成cache文件，注意！！！更改max_seq_len之后需要去掉--cache重新运行一遍
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -weight_name model.pkl \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                -max_seq_len 200 \
                -vocab_dict_path data/assets/page2idx.json \
                -vocab_size 10000 \
                -backbone transformer \
                -embedding_dim 280 \
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

# 之后的运行可以指定cache参数。
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -weight_name model.pkl \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                --use_cache \
                -max_seq_len 200 \
                -vocab_dict_path data/assets/page2idx.json \
                -vocab_size 10000 \
                -backbone transformer \
                -embedding_dim 280 \
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

### 按照user_id切分用户访问页面

只要加上-data_type user即可(最好调一下max_seq_len参数调大一些，当然也不是说这么做就好，只是直觉上感觉，具体这么做有没有效果，调参决定)，注意！！！更改max_seq_len和data_type之后都需要去掉--cache重新运行一遍。

第一次运行不加上--use_cache选项以生成cache，之后加上--use_cache选项。
```bash
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -weight_name model.pkl \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                --use_cache \
                -data_type pageuser \
                -max_seq_len 300 \
                -vocab_dict_path data/assets/page2idx.json \
                -vocab_size 10000 \
                -backbone transformer \
                -embedding_dim 380 \
                -ffn_num_hiddens 300 \
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

### 按照单词建立词典，而后建立向量

第一次运行不加上--use_cache选项以生成cache，之后加上--use_cache选项。
```bash
python train.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -weight_name model.pkl \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                --use_cache \
                -data_type worduser \
                -max_seq_len 400 \
                -vocab_dict_path data/assets/word2idx.json \
                -vocab_size 10000 \
                -backbone transformer \
                -embedding_dim 512 \
                -ffn_num_hiddens 400 \
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

## Test


注意！！！用什么样子的参数训练，就要用什么u样子的参数测试。
```bash
python test.py -dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                -weight_name model.pkl \
                -file_name_abnormal feedback.csv \
                -file_name_normal normal.csv \
                --use_cache \
                -max_seq_len 200 \
                -vocab_dict_path data/assets/page2idx.json \
                -vocab_size 10000 \
                -backbone transformer \
                -embedding_dim 280 \
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

## Reference

["Explainable Deep Few-shot Anomaly Detection with Deviation Networks"](https://arxiv.org/abs/2108.00462)