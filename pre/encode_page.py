"""使用feedback的用户log，给所有页面进行编码，同时获取session中最大页面的长度
"""
import os
import pandas as pd
import argparse
import time
import sys
import json

from wash_pagename import *

cur_login_user = os.getlogin()
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
cur_abs_working_directory = os.path.abspath(
    "/home/{}/Source/deviation-network-fliggy/".format(cur_login_user))  # 设置当前项目的工作目录
os.chdir(cur_abs_working_directory)
print("current working directory:", os.getcwd())
cur_time = '2023-01-20-21-57-52'

sys.path.append("./")

parser = argparse.ArgumentParser(description='')
parser.add_argument('-infile_directory', type=str,
                    default="data/datasets/", help='input file directory')
args = parser.parse_args()


def tmp_prepare_data(file_name: str):
    data_path = os.path.join(args.infile_directory, file_name)
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)
    return df


def generate_pagename_cnt():
    """使用feedback和normal的用户log，计算每一个page_name的数量
    """
    normal = tmp_prepare_data("normal.csv")
    feedback = tmp_prepare_data("feedback.csv")
    all = pd.concat([normal, feedback], axis=0).drop(columns=['unique_id']).reset_index(
        drop=True).reset_index().rename(columns={"index": "unique_id"})

    all_page_num = all.groupby('page_name')[
        'unique_id'].count().sort_values(ascending=False)
    page2num = dict(
        zip(all_page_num.index, [int(value) for value in all_page_num.values]))
    json.dump(page2num, open(
        f"pre/data/page2num-origin-simulate.json", "w"), indent=4)


def generate_lastword_dict(page2num):
    """生成最后一个单词的字典, 词频统计后手动生成lastword_dict
    Args:
        page2num (dict): page_name -> num
    Returns:
        None
    """
    words2freq = {}
    for page, num in page2num.items():
        # 按照非字母数字下划线-进行分词
        words = [word for word in re.split(
            r'[^a-zA-Z0-9_-]', page.lower()) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
        # 按照非字母数字下划线进行分词
        words = [word for word in re.split(
            r'[^a-zA-Z0-9]', page.lower()) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
        # 按照非字母数字进行分词
        words = [word for word in re.split(
            r'[^a-zA-Z0-9_]', page.lower()) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
                
    # >=50的词作为lastword_dict的成员
    lastword_list_freq = [word for word in words2freq.keys()
                          if words2freq[word] >= 50 and word]

    # 手动查找单词
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    from nltk.corpus import wordnet
    lastword_list_nltk = []
    for word in words2freq.keys():
        if wordnet.synsets(word) != []:  # 是一个单词
            lastword_list_nltk.append(word)
    lastword_list_nltk = [word for word in lastword_list_nltk if word]

    # 载入旧的列表
    lastword_list_old = json.load(open("pre/assets/lastword_dict.json", "r"))

    # 更新列表
    lastword_list = list(
        set(lastword_list_freq + lastword_list_nltk + lastword_list_old))
    json.dump(lastword_list, open(
        f"pre/assets/lastword_dict.json", "w"), indent=4)


def wash_pagename(page2num_path: str):
    """
    Args:
        page2num_path (str): page_name到数量的映射
    Returns:
        afterwash_dict (dict): 清洗过的page_name到数量的映射
        washfailed_dict (dict): 清洗失败的page_name到数量的映射
    """
    page2num = json.load(open(page2num_path, 'r'))
    url_list, url_cnt = list(page2num.keys()), list(page2num.values())
    url_list = list(map(preprocess, url_list))
    # 经过预处理一些url可能已经重合，进行合并
    url_list, url_cnt = merge_url(url_list, url_cnt)

    safe_list, safe_num = [], [] # 已经处理过不再处理的url

    # 根据url本身做过滤（是否是url、结尾的拓展名、是否在lastword_dict中）
    index_list = list(map(filter_by_url, url_list))
    safe_list.extend(np.array(url_list)[index_list].tolist())
    safe_num.extend(np.array(url_cnt)[index_list].tolist())
    url_list = np.array(url_list)[~np.array(index_list)].tolist()
    url_cnt = np.array(url_cnt)[~np.array(index_list)].tolist()

    # 按照最后一个词出现的频率进行过滤，生成freq_lastword_dict
    index_list = filter_by_lastword_frequency(url_list, url_cnt)
    safe_list.extend(np.array(url_list)[index_list].tolist())
    safe_num.extend(np.array(url_cnt)[index_list].tolist())
    url_list = np.array(url_list)[~np.array(index_list)].tolist()
    url_cnt = np.array(url_cnt)[~np.array(index_list)].tolist()

    # 按照url出现的频率进行过滤，生成freq_url_dict
    index_list = filter_by_url_frequency(url_list, url_cnt)
    safe_list.extend(np.array(url_list)[index_list].tolist())
    safe_num.extend(np.array(url_cnt)[index_list].tolist())
    url_list = np.array(url_list)[~np.array(index_list)].tolist()
    url_cnt = np.array(url_cnt)[~np.array(index_list)].tolist()

    safe_list, safe_num, url_list, url_cnt = filter_by_special(
        safe_list, safe_num, url_list, url_cnt)  # 这个操作放后面，特殊情况特殊处理

    url_list = list(map(process_by_force, url_list))
    url_list, url_cnt = merge_url(url_list, url_cnt)  # 经过预处理一些url可能已经重合，进行合并
    index_list = [True for _ in range(len(url_list))]
    safe_list.extend(np.array(url_list)[index_list].tolist())
    safe_num.extend(np.array(url_cnt)[index_list].tolist())
    url_list = np.array(url_list)[~np.array(index_list)].tolist()
    url_cnt = np.array(url_cnt)[~np.array(index_list)].tolist()

    afterwash_dict = dict(zip(safe_list, safe_num))
    # json.dump(afterwash_dict, open('pre/data/afterwash_dict.json', 'w'), indent=4)
    washfailed_dict = dict(zip(url_list, url_cnt))
    # json.dump(washfailed_dict, open('pre/data/washfailed_dict.json', 'w'), indent=4)
    return afterwash_dict, washfailed_dict


def encode_page(pagename_cnt_path: str):
    """使用清洗过的page_name文件，给所有页面进行编码
    Args:
        pagename_cnt_path (str): 清洗过的page_name文件
    Returns:
        page2idx (dict): page_name到idx的映射
        idx2page (dict): idx到page_name的映射
    """
    all_page_num = json.load(open(pagename_cnt_path, 'r'))
    all_page = list(all_page_num.keys())
    page2idx = {
        '<eos>': 0,
        '<unk>': 1,  # 未知页面
        '<pad>': 2
    }
    page2idx.update(dict(zip(all_page, range(3, len(all_page) + 3))))
    json.dump(page2idx, open(f"pre/assets/page2idx.json", "w"), indent=4)
    idx2page = dict(zip(range(len(all_page) + 3),
                    ['<eos>', '<unk>', '<pad>'] + list(all_page)))
    json.dump(idx2page, open(f"pre/assets/idx2page.json", "w"), indent=4)


if __name__ == "__main__":
    # generate_pagename_cnt()

    # 一周的page2num的名字
    page2num_paths = [
        'pre/data/page2num-origin-2023-01-20-21-57-52.json',
        'pre/data/page2num-origin-simulate.json',
        # 'pre/data/page2num-origin-2023-01-20-21-57-52.json',
        # 'pre/data/page2num-origin-2023-01-20-21-57-52.json',
        # 'pre/data/page2num-origin-2023-01-20-21-57-52.json',
        # 'pre/data/page2num-origin-2023-01-20-21-57-52.json',
        # 'pre/data/page2num-origin-2023-01-20-21-57-52.json',
    ]

    page2num_jsons = [json.load(open(page2num_path, 'r'))
                      for page2num_path in page2num_paths]
    page2num = {}
    for page2num_json in page2num_jsons:
        for page, num in page2num_json.items():
            if page in page2num:
                page2num[page] += num
            else:
                page2num[page] = num
    page2num_merge_name = f'pre/data/page2num-merge-{cur_time}.json'
    json.dump(page2num, open(page2num_merge_name, "w"), indent=4)

    generate_lastword_dict(page2num)

    afterwash_dict, washfailed_dict = wash_pagename(
        page2num_path=page2num_merge_name)
    json.dump(afterwash_dict, open(
        'pre/data/page2num_afterwash.json', 'w'), indent=4)
    json.dump(washfailed_dict, open(
        'pre/data/page2num_failedwash.json', 'w'), indent=4)

    encode_page(pagename_cnt_path='pre/data/page2num_afterwash.json')
