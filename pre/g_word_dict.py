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


def generate_word_list(page2num):
    """生成最后一个单词的字典, 词频统计后手动生成lastword_dict
    Args:
        page2num (dict): page_name -> num
    Returns:
        lastword_dict (list): lastword
    """
    words2freq = {}
    for page, num in page2num.items():
        # 按照非字母数字进行分词
        words = [word.lower() for word in re.split(
            r'[^a-zA-Z0-9]', page) if word]
        for word in words:
            if word in words2freq:
                words2freq[word] += num
            else:
                words2freq[word] = num

    # >=10的词作为word_dict的成员
    words_list_freq = [word for word in words2freq.keys()
                       if words2freq[word] >= 10 and word]

    # 手动查找单词
    word_list_nltk = []
    for word in words2freq.keys():
        if wordnet.synsets(word) != []:  # 是一个单词
            word_list_nltk.append(word)
    word_list_nltk = [word for word in word_list_nltk if word]

    # 更新列表
    word_dict = list(
        set(words_list_freq + word_list_nltk))

    return word_dict


def encode_word(word_list):
    """使用清洗过的page_name文件，给所有页面进行编码
    Args:
        word_dict (list): 字典里的单词
    Returns:
        word2idx (dict): word到idx的映射
        idx2word (dict): idx到word的映射
    """
    all_words = word_list
    word2idx = {
        '<eos>': 0,
        '<unk>': 1,  # 未知页面
        '<pad>': 2
    }
    word2idx.update(dict(zip(all_words, range(3, len(all_words) + 3))))

    idx2word = dict(zip(range(len(all_words) + 3),
                    ['<eos>', '<unk>', '<pad>'] + list(all_words)))
    return word2idx, idx2word


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

    word_list = generate_word_list(page2num)

    output_dir = pre_args.page2num_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(word_list, open(os.path.join(output_dir,
                                           "word_dict.json"), "w"), indent=4)

    word2idx, idx2word = encode_word(word_list)
    output_dir_word_dict = pre_args.output_dir_word_dict
    if not os.path.exists(output_dir_word_dict):
        os.makedirs(output_dir_word_dict)
    json.dump(word2idx, open(os.path.join(output_dir_word_dict,
                                          "word2idx.json"), "w"), indent=4)
    json.dump(idx2word, open(os.path.join(output_dir_word_dict,
                                          "idx2word.json"), "w"), indent=4)
