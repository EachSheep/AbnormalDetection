
import json
import re
import os
from pre_argparser import pre_args

# lastword_dict: 一定是关键词的词典
if not os.path.exists(pre_args.lastword_dict_dir):
    os.makedirs(pre_args.lastword_dict_dir)
if not os.path.exists(os.path.join(pre_args.lastword_dict_dir, pre_args.lastword_dict_name)):
    with open(os.path.join(pre_args.lastword_dict_dir, pre_args.lastword_dict_name), 'w') as f:
        json.dump([], f)
lastword_dict = json.load(open(os.path.join(
    pre_args.lastword_dict_dir, pre_args.lastword_dict_name), 'r'))  # 读取safe_dict

last_dict = list(set(lastword_dict))


def url_preprocess(url):
    """处理url
    """
    if re.match(r'^https?:\/\/', url):
        # 末尾必须是非字母数字
        url_lastword1 = re.search(r'[^a-zA-Z0-9]([a-zA-Z0-9]+)$', url)
        url_lastword2 = re.search(r'[^a-zA-Z0-9_]([a-zA-Z0-9_]+)$', url)
        url_lastword3 = re.search(r'[^a-zA-Z0-9-]([a-zA-Z0-9-]+)$', url)
        url_lastword4 = re.search(r'[^a-zA-Z0-9_-]([a-zA-Z0-9_-]+)$', url)
        if url_lastword1 and url_lastword1.group()[1:] in last_dict or url_lastword2 and url_lastword2.group()[1:] in last_dict or url_lastword3 and url_lastword3.group()[1:] in last_dict or url_lastword4 and url_lastword4.group()[1:] in last_dict:
            return url
        return '/'.join(url.split('/')[: -1])
    else:
        # 正则表达式匹配非url
        return url
