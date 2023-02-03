
import json
import re
import os

# lastword_dict: 一定是关键词的词典
if not os.path.exists(os.path.join(os.path.dirname(__file__), '../data/assets/')):
    os.makedirs(os.path.join(os.path.dirname(__file__), '../data/assets/'))
if not os.path.exists(os.path.join(os.path.dirname(__file__), '../data/assets/lastword_dict.json')):
    with open(os.path.join(os.path.dirname(__file__), '../data/assets/lastword_dict.json'), 'w') as f:
        json.dump([], f)
lastword_dict = json.load(open(os.path.join(os.path.dirname(
    __file__), '../data/assets/lastword_dict.json'), 'r'))  # 读取safe_dict

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
        return '/'.join(url.split('/')[:-1])
    else:
        return url