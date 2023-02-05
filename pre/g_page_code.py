"""使用feedback的用户log，给所有页面进行编码，同时获取session中最大页面的长度
"""
import os
import json
import nltk

from pre_argparser import pre_args
from UrlPreprocess import url_preprocess

nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess(page2num: dict):
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


def encode_page(afterwash_dict):
    """使用清洗过的page_name文件，给所有页面进行编码
    Args:
        afterwash_dict (dict): 清洗过的page_name文件
    Returns:
        page2idx (dict): page_name到idx的映射
        idx2page (dict): idx到page_name的映射
    """
    all_page_num = afterwash_dict
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

    afterwash_dict = preprocess(page2num)
    page2num_afterwash_path = os.path.join(
        pre_args.page2num_dir, "page2num_afterwash_encodepage.json")
    json.dump(afterwash_dict, open(
        page2num_afterwash_path, 'w'), indent=4)

    page2idx, idx2page = encode_page(afterwash_dict)
    page2idx_path = os.path.join(
        pre_args.output_dir_lastword_dict, "page2idx.json")
    idx2page_path = os.path.join(
        pre_args.output_dir_lastword_dict, "idx2page.json")
    json.dump(page2idx, open(page2idx_path, "w"), indent=4)
    json.dump(idx2page, open(idx2page_path, "w"), indent=4)
