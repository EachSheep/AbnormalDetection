# preprocess

## run prepreprocess.py

```bash
python prepreprocess.py -in_dir ../data/datasets/ \
                        -feedback_names feedback.csv  \
                        -normal_names normal.csv  \
                        -output_dir ../data/ \
                        -page2num_names_o page2num-simulate.json
```

## run encode_page.py

```bash
python encode_page.py -page2num_dir ../data/ \
                      -page2num_names_i page2num-1.json \
                      -page2num_merge_name page2num-merge.json \
                      -page2num_afterwash_name page2num-afterwash.json \
                      --simulate
```

## run draw.py

```bash
python draw.py -in_dir ../data/ \
               -feedback_names feedback.csv  \
               -normal_names normal.csv
```