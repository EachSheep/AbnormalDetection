# 预预处理

## run prepreprocess.py

```bash
python prepreprocess.py -in_dir_prepre ../data/datasets/ \
                        -feedback_names_prepre feedback.csv  \
                        -normal_names_prepre normal.csv  \
                        -output_dir_prepre ../data/prepreprocess/
```


# 生成编码

根据预预处理得到的文件生成页面的编码

## run g_page2num.py

首先生成page2num.json文件（已经生成，不用管）

```bash
python g_page2num.py -in_dir_gen ../data/prepreprocess/ \
                     -feedback_names_gen feedback.csv  \
                     -normal_names_gen normal.csv  \
                     -output_dir_gen ../data/page2nums/ \
                     -page2num_names_gen page2num-simulate.json
```

## run g_lastword_dict.py

根据page2num-simulate.json生成lastword_dict.json

```bash
python g_lastword_dict.py -page2num_dir ../data/page2nums/ \
                        -page2num_names page2num-1.json \
                        --simulate
```

## run g_code.py

然后根据page2num.json文件生成页面的编码

```bash
python g_code.py -lastword_dict_dir ../data/page2nums/ \
                -lastword_dict_name lastword_dict.json \
                -output_dir_lastword_dict ../data/assets/ \
                --simulate
```

# 跑训练之前

需要对训练和测试数据做和生成字典同样的预处理

## run preprocess.py

```bash
python preprocess.py -in_dir_pre ../data/prepreprocess/ \
                     -feedback_names_pre feedback.csv  \
                     -normal_names_pre normal.csv  \
                     -output_dir_pre ../data/preprocess/
```



## run draw.py

```bash
python draw.py -in_dir ../data/prepreprocess \
               -feedback_names feedback.csv  \
               -normal_names normal.csv
```