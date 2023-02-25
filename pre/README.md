# 预预处理

## run prepreprocess.py

真实跑的时候（xxx改成文件所在目录，两个文件分别重命名为feedback.csv和normal.csv）：
```bash
python prepreprocess.py -in_dir_prepre xxx \
                        -feedback_names_prepre feedback.csv  \
                        -normal_names_prepre normal.csv  \
                        -output_dir_prepre ../experiment/prepreprocess/
```

模拟时候：
```bash
python prepreprocess.py -in_dir_prepre ../experiment/datasets/ \
                        -feedback_names_prepre feedback.csv  \
                        -normal_names_prepre normal.csv  \
                        -output_dir_prepre ../experiment/prepreprocess/
```


# 生成编码

根据预预处理得到的文件生成页面的编码

## run g_page2num.py

首先生成page2num.json文件，**注意：该文件已经生成，不用管下面的命令！！！**

真实跑的时候：
```bash
python g_page2num.py -in_dir_gen ../experiment/prepreprocess/ \
                     -feedback_names_gen feedback.csv  \
                     -normal_names_gen normal.csv  \
                     -output_dir_gen ../experiment/page2nums/ \
                     -page2num_names_gen page2num-1.json
```

模拟时候：
```bash
python g_page2num.py -in_dir_gen ../experiment/prepreprocess/ \
                     -feedback_names_gen feedback.csv  \
                     -normal_names_gen normal.csv  \
                     -output_dir_gen ../experiment/page2nums/ \
                     -page2num_names_gen page2num-simulate.json
```

## run g_lastword_dict.py

**注意：该文件已经生成，不用管下面的命令！！！**

**给页面进行编码：**

真实跑的时候：
```bash
python g_lastword_dict.py -page2num_dir ../experiment/page2nums/ \
                        -page2num_names page2num-1.json
```

模拟时候：根据 page2num-1.json和page2num-simulate.json生成lastword_dict.json，
```bash
python g_lastword_dict.py -page2num_dir ../experiment/page2nums/ \
                        -page2num_names page2num-1.json \
                        --simulate
```

**给单词进行编码：**

真实跑的时候：
```bash
python g_word_dict.py -page2num_dir ../experiment/page2nums/ \
                        -page2num_names page2num-1.json
```

模拟时候：根据 page2num-1.json和page2num-simulate.json生成word_dict.json，
```bash
python g_word_dict.py -page2num_dir ../experiment/page2nums/ \
                      -page2num_names page2num-1.json \
                      --simulate \
                      -output_dir_word_dict ../experiment/assets/
```

## run g_page_code.py

**给页面进行编码：**

然后根据lastword_dict.json文件生成页面的编码.

真实跑的时候：
```bash
python g_page_code.py -lastword_dict_dir ../experiment/page2nums/ \
                -lastword_dict_name lastword_dict.json \
                -output_dir_lastword_dict ../experiment/assets/
```

模拟时候：
```bash
python g_page_code.py -lastword_dict_dir ../experiment/page2nums/ \
                -lastword_dict_name lastword_dict.json \
                -output_dir_lastword_dict ../experiment/assets/ \
                --simulate
```

# 跑训练之前

需要对训练和测试数据做和生成字典同样的预处理

## run preprocess.py

```bash
python preprocess.py -in_dir_pre ../experiment/prepreprocess/ \
                     -feedback_names_pre feedback.csv  \
                     -normal_names_pre normal.csv  \
                     -output_dir_pre ../experiment/preprocess/
```

# 其它

## run draw.py：绘图

```bash
python draw.py -in_dir ../experiment/prepreprocess \
               -feedback_names feedback.csv  \
               -normal_names normal.csv
```

## run g_lowercase2uppercase.py

真实跑的时候（xxx改成文件所在目录，两个文件分别重命名为feedback.csv和normal.csv）：
```bash
python g_lowercase2uppercase.py -in_dir_prepre xxx \
                        -feedback_names_prepre feedback.csv  \
                        -normal_names_prepre normal.csv  \
                        -output_dir_prepre ../experiment/assets/
```

模拟时候：
```bash
python g_lowercase2uppercase.py -in_dir_prepre ../experiment/datasets/ \
                        -feedback_names_prepre feedback.csv  \
                        -normal_names_prepre normal.csv  \
                        -output_dir_prepre ../experiment/assets/
```
