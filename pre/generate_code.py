"""使用feedback的用户log，给所有页面进行编码，同时获取session中最大页面的长度
"""
import os
import json
import nltk
import re
from nltk.corpus import wordnet

from pre_argparser import pre_args
from UrlPreprocess import url_preprocess

nltk.download('wordnet')
nltk.download('omw-1.4')


def generate_lastword_list(page2num):
    """生成最后一个单词的字典, 词频统计后手动生成lastword_dict
    Args:
        page2num (dict): page_name -> num
    Returns:
        lastword_dict (list): lastword
    """
    words2freq = {}
    for page, num in page2num.items():
        # 按照非字母数字下划线-进行分词
        words = [word for word in re.split(
            r'[^a-zA-Z0-9_-]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
        # 按照非字母数字-进行分词
        words = [word for word in re.split(
            r'[^a-zA-Z0-9-]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
        # 按照非字母数字下划线进行分词
        words = [word for word in re.split(
            r'[^a-zA-Z0-9_]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
        # 按照非字母数字进行分词
        words = [word for word in re.split(
            r'[^a-zA-Z0-9]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num

    # >=50的词作为lastword_dict的成员
    lastword_list_freq = [word for word in words2freq.keys()
                          if words2freq[word] >= 50 and word]

    # 手动查找单词
    lastword_list_nltk = []
    for word in words2freq.keys():
        if wordnet.synsets(word) != []:  # 是一个单词
            lastword_list_nltk.append(word)
    lastword_list_nltk = [word for word in lastword_list_nltk if word]

    # 更新列表
    lastword_list = list(
        set(lastword_list_freq + lastword_list_nltk))

    return lastword_list


def wash_pagename(page2num: dict):
    """
    Args:
        page2num (dict): page_name到数量的映射
    Returns:
        afterwash_dict (dict): 清洗过的page_name到数量的映射
    """
    keys = list(page2num.keys())
    values = list(page2num.values())
    keys = list(map(url_preprocess, keys))
    afterwash_dict = dict(zip(keys, values))

    def merge(afterwash_dict):
        tmp = {}
        for key, value in afterwash_dict.items():
            tmp[key] = tmp.get(key, 0) + value
        return tmp
    afterwash_dict = merge(afterwash_dict)

    return afterwash_dict


def encode_page(pagename_cnt_path):
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

    idx2page = dict(zip(range(len(all_page) + 3),
                    ['<eos>', '<unk>', '<pad>'] + list(all_page)))
    return page2idx, idx2page


if __name__ == "__main__":
    page2num_paths = [os.path.join(pre_args.page2num_dir, page2num_name)
                      for page2num_name in pre_args.page2num_names]
    page2num_jsons = [json.load(open(page2num_path, 'r'))
                      for page2num_path in page2num_paths]
    page2num = {}
    for page2num_json in page2num_jsons:
        for page, num in page2num_json.items():
            if page in page2num:
                page2num[page] += num
            else:
                page2num[page] = num
    page2num_merge_path = os.path.join(
        pre_args.page2num_dir, pre_args.page2num_merge_name)
    json.dump(page2num, open(page2num_merge_path, "w"), indent=4)

    page2num = json.load(
        open(page2num_merge_path, 'r'))  # 根据频率和是否是单词生成词典
    lastword_list = generate_lastword_list(page2num)
    assets_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../data/assets")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    json.dump(lastword_list, open(os.path.join(assets_dir,
                                               "lastword_dict.json"), "w"), indent=4)

    afterwash_dict = wash_pagename(page2num)
    afterwash_dict = page2num
    page2num_afterwash_path = os.path.join(
        pre_args.page2num_dir, pre_args.page2num_afterwash_name)
    json.dump(afterwash_dict, open(
        page2num_afterwash_path, 'w'), indent=4)

    page2idx, idx2page = encode_page(pagename_cnt_path=page2num_afterwash_path)
    page2idx_path = os.path.join(
        assets_dir, "page2idx.json")
    idx2page_path = os.path.join(
        assets_dir, "idx2page.json")
    json.dump(page2idx, open(page2idx_path, "w"), indent=4)
    json.dump(idx2page, open(idx2page_path, "w"), indent=4)
