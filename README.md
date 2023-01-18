# Abnormal Detection

## Setup 

Install with `pip install -r requirements.txt`.

Or follow instructions in requirements.txt to install.

## Running

```bash
python train.py --dataset_root=/home/hiyoungshen/Source/deviation-network-fliggy/data/datasets/ \
                --file_name_abnormal feedback.csv \
                --file_name_normal normal.csv \
                --filter_num 10 \
                --vocab_dict_path pre/data/page2id-2023-01-18-12-12-23.json \
                --max_seq_len 100 \
                --vocab_size 10000 \
                --embedding_dim 300 \
                --hidden_dim 512 \
                --lr 0.0002 \
                --epochs 50 \
                --batch_size 48 \
```


## Reference

["Explainable Deep Few-shot Anomaly Detection with Deviation Networks"](https://arxiv.org/abs/2108.00462)