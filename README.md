# Abnormal Detection

## Setup 

Install with `pip install -r requirements.txt`.

Or follow instructions in requirements.txt to install.

## Running

### train 

训练的时候

```bash
python train.py --dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/preprocess/ \
                --file_name_abnormal feedback.csv \
                --file_name_normal normal.csv \
                --max_seq_len 200 \
                --vocab_dict_path data/assets/page2idx.json \
                --vocab_size 10000 \
                --embedding_dim 300 \
                --hidden_dim 128 \
                --lr 0.0002 \
                --epochs 50 \
                --batch_size 128 \
                > experiment/log.txt
```

### test

```bash
python test.py --dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/ \
                --file_name_abnormal feedback.csv \
                --file_name_normal normal.csv \
                --min_seq_len 10 \
                --max_seq_len 200 \
                --vocab_dict_path pre/data/page2idx.json > experiment/log.txt
```

## Reference

["Explainable Deep Few-shot Anomaly Detection with Deviation Networks"](https://arxiv.org/abs/2108.00462)