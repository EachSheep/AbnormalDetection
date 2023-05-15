"""使用feedback的用户log，给所有页面进行编码，同时获取session中最大页面的长度
"""
import os
import json
import nltk
import re
from nltk.corpus import wordnet

from pre_argparser import pre_args

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
    lastwords2freq = {}
    for page, num in page2num.items():
        # 按照非字母数字下划线-进行分词
        words = [word.lower() for word in re.split(
            r'[^a-zA-Z0-9_-]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
        # 按照非字母数字-进行分词
        words = [word.lower() for word in re.split(
            r'[^a-zA-Z0-9-]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
        # 按照非字母数字下划线进行分词
        words = [word.lower() for word in re.split(
            r'[^a-zA-Z0-9_]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num
        # 按照非字母数字进行分词
        words = [word.lower() for word in re.split(
            r'[^a-zA-Z0-9]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num

        if re.match(r'^https?:\/\/', page):
            _1 = re.search(r'[^a-zA-Z0-9_-]([a-zA-Z0-9_-]*)$', page)
            _2 = re.search(r'[^a-zA-Z0-9_]([a-zA-Z0-9_]*)$', page)
            _3 = re.search(r'[^a-zA-Z0-9-]([a-zA-Z0-9-]*)$', page)
            _4 = re.search(r'[^a-zA-Z0-9]([a-zA-Z0-9]*)$', page)
            if _1:
                word = _1.group()[1:].lower()
                if word in lastwords2freq:
                    lastwords2freq[word] += num
                else:
                    lastwords2freq[word] = num
            if _2:
                word = _2.group()[1:].lower()
                if word in lastwords2freq:
                    lastwords2freq[word] += num
                else:
                    lastwords2freq[word] = num
            if _3:
                word = _3.group()[1:].lower()
                if word in lastwords2freq:
                    lastwords2freq[word] += num
                else:
                    lastwords2freq[word] = num
            if _4:
                word = _4.group()[1:].lower()
                if word in lastwords2freq:
                    lastwords2freq[word] += num
                else:
                    lastwords2freq[word] = num

    # >=50的词作为lastword_dict的成员
    words_list_freq = [word for word in words2freq.keys()
                       if words2freq[word] >= 50 and word]
    lastwords_list_freq = [word for word in lastwords2freq.keys()
                           if lastwords2freq[word] >= 10 and word]

    # 手动查找单词
    lastword_list_nltk = []
    for word in words2freq.keys():
        if wordnet.synsets(word) != []:  # 是一个单词
            lastword_list_nltk.append(word)
    lastword_list_nltk = [word for word in lastword_list_nltk if word]

    # 更新列表
    lastword_list = list(
        set(words_list_freq + lastword_list_nltk))

    return lastword_list


if __name__ == "__main__":
    page2num_paths = [os.path.join(pre_args.page2num_dir, page2num_name)
                      for page2num_name in pre_args.page2num_names]
    page2num_jsons = [json.load(open(page2num_path, 'r'))
                      for page2num_path in page2num_paths]
    output_dir = pre_args.output_dir_lastword
    lastword_dict_name = pre_args.lastword_dict_names[0]

    page2num = {}
    for page2num_json in page2num_jsons:
        for page, num in page2num_json.items():
            if page in page2num:
                page2num[page] += num
            else:
                page2num[page] = num

    lastword_list = generate_lastword_list(page2num)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(lastword_list, open(os.path.join(output_dir,
                                               lastword_dict_name), "w"), indent=4)
