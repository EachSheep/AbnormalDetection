"""对page_num进行清洗
"""
import json
import re
import numpy as np
import os


def preprocess(url):
    """预处理
    Args:
        url (str): 待处理的url
    Returns:
        url (str): 处理后的url
    # 筛除局域网内广播信息
    # index_list = [True if not re.match(r'^https?:\/\/(192\.168|10|172\.1[6-9]|172\.2[0-9]|172\.3[0-1])\.', url) else False for url in url_list]
    """
    url = url.lower()  # 大写变小写
    # 去除url中的汉字, 去除所有的url中文编码, 去除所有逗号, 去除所有~
    url = re.sub(r'[\u4e00-\u9fa5]|%[a-f\d]{2}|~|,', '', url)
    # "https://m.amap.com/navigation/carmap/&saddr=121.834638%2c29.847424%2c%e6%88%91%e7%9a%84%e4%bd%8d%e7%bd%ae&daddr=121.51234007%2c29.84995423%2c%e5%ae%81%e6%b3%a2%e5%ae%a2%e8%bf%90%e4%b8%ad%e5%bf%83%e5%9c%b0%e9%93%81%e7%ab%99c%e5%8f%a3"
    # 类似这样的url变成：https://m.amap.com/navigation/carmap/&saddr=daddr=
    if '=' in url:  # =到下一个&，或者=到最后的字符去除
        url = re.sub(r'=.*&|=.*$', '=', url)
    if len(url) > 0 and url[-1] == '=':
        url = url[:-1]
    url = re.sub(r'/+$', '', url)  # 末尾不能以/结尾
    return url


def merge_url(url_list, url_num):
    """合并url
    Args:
        url_list (list): 待合并的url列表
        url_num (list): 合并后的url数量
    Returns:
        url_list (list): 合并后的url列表
        url_num (list): 合并后的url数量
    """
    tmp = {}
    for i in range(len(url_list)):
        tmp[url_list[i]] = tmp.get(url_list[i], 0) + url_num[i]
    return list(tmp.keys()), list(tmp.values())


def filter_by_ifurl(url):
    """根据是否是url过滤
    Args:
        url (str): 待过滤的url
    Returns:
        filter_or_not (bool) : 是否被过滤掉，是为True, 否为False
    """
    if re.match(r'^https?:\/\/', url):
        return False
    else:
        return True


extension_of_filename = set([extension.lower() for extension in json.load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/extension_of_filename.json'), 'r'))])  # 以各种拓展名结尾的


def filter_by_extension(url):
    """根据拓展名过滤url
    Args:
        url (str): 待过滤的url
    Returns:
        filter_or_not (bool) : 是否被过滤掉，是为True, 否为False
    """
    new_url = url.split('/')[-1]
    extension_in_url = re.search(r'[^a-zA-Z0-9]([a-zA-Z0-9_]+)$', new_url)
    if extension_in_url and extension_in_url.group()[1:] in extension_of_filename:
        return True
    else:
        return False


# lastword_dict: 一定是关键词的词典
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'assets/')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'assets/'))
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'assets/lastword_dict.json')):
    with open(os.path.join(os.path.dirname(__file__), 'assets/lastword_dict.json'), 'w') as f:
        json.dump({}, f)
lastword_dict = json.load(open(os.path.join(os.path.dirname(
    __file__), 'assets/lastword_dict.json'), 'r'))  # 读取safe_dict
lastword_dict = [word.lower() for word in lastword_dict]
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'assets/')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'assets/'))
if not os.path.exists('pre/assets/freq_lastword_dict.json'):
    with open(os.path.join(os.path.dirname(__file__), 'assets/freq_lastword_dict.json'), 'w') as f:
        json.dump({}, f)
# 读取freq_lastword_dict
freq_lastword_dict = json.load(open(os.path.join(os.path.dirname(
    __file__), 'assets/freq_lastword_dict.json'), 'r'))
freq_lastword_dict = [word.lower() for word in freq_lastword_dict]

last_dict = list(set(lastword_dict + freq_lastword_dict))


def filter_by_dict(url):
    """根据词典过滤url，以词典中的词结尾的url为安全url，生成字典时候要用到
    Args:
        url (str): 待过滤的url
    Returns:
        filter_or_not (bool) : 是否被过滤掉，是为True, 否为False
    """
    # 末尾必须是非字母数字
    url_lastword = re.search(r'[^a-zA-Z0-9]([a-zA-Z0-9_]+)$', url)
    if url_lastword and url_lastword.group()[1:] in last_dict:
        return True
    else:
        return False


def filter_by_url(url):
    """过滤url
    Args:
        url (str): 待过滤的url
    Returns:
        filter_or_not (bool) : 是否被过滤掉，是为True, 否为False
    """
    return filter_by_ifurl(url) or filter_by_extension(url) or filter_by_dict(url)


if not os.path.exists(os.path.join(os.path.dirname(__file__), 'assets/')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'assets/'))
if not os.path.exists('pre/assets/freq_url_dict.json'):
    with open(os.path.join(os.path.dirname(__file__), 'assets/freq_url_dict.json'), 'w') as f:
        json.dump({}, f)
# 读取freq_url_dict
freq_url_dict = json.load(open(os.path.join(os.path.dirname(
    __file__), 'assets/freq_url_dict.json'), 'r'))
freq_url_dict = [word.lower() for word in freq_url_dict]


def filter_by_freq_url(url):
    """过滤url
    Args:
        url (str): 待过滤的url
    Returns:
        filter_or_not (bool) : 是否被过滤掉，是为True, 否为False
    """
    if url in freq_url_dict:
        return True
    else:
        return False


def filter_by_url_frequency(url_list, url_num):
    """按照url的频率过滤url，同时更新freq_url_dict
    freq_url_dict: 自动根据当天的频率生成的词典，最多保留10000条，超了过后，旧的词会被删除

    Args:
        url_list (list): 待过滤的url
        url_num (list): 待过滤的url的出现次数
    Returns:
        index_list (list): 安全的url的索引
    """
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets/')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    freq_url_path = os.path.join(assets_dir, 'freq_url_dict.json')
    if not os.path.exists(freq_url_path):
        with open(freq_url_path, 'w') as f:
            json.dump({}, f)

    url_dict = [url_list[i]
                for i in range(len(url_list)) if url_num[i] >= 10]  # 筛选出出现次数多的url

    # 读取freq_dict
    freq_url_dict = json.load(open(freq_url_path, 'r'))
    freq_url_dict = [cur_url.lower() for cur_url in freq_url_dict]
    freq_url_dict = freq_url_dict + url_dict
    freq_url_dict = sorted(set(freq_url_dict), key=freq_url_dict.index)  # 稳定去重
    if len(freq_url_dict) > 10000:
        freq_url_dict = freq_url_dict[-10000:]
    json.dump(freq_url_dict, open(freq_url_path, 'w'), indent=4)  # 更新词典至文件

    index_list = np.array(  # 以词典中的词结尾的url为安全url
        [True if cur_url in freq_url_dict else False for cur_url in url_list])

    return index_list


def filter_by_lastword_frequency(url_list, url_num):
    """按照url最后一个/后面的词的频率过滤url，同时更新freq_lastword_dict
    freq_lastword_dict: 自动根据当天的频率生成的词典，最多保留10000条，超了过后，旧的词会被删除

    Args:
        url_list (list): 待过滤的url
        url_num (list): 待过滤的url的出现次数
    Returns:
        index_list (list): 安全的url的索引
    """
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets/')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    freq_lastword_path = os.path.join(assets_dir, 'freq_lastword_dict.json')
    if not os.path.exists(freq_lastword_path):
        with open(freq_lastword_path, 'w') as f:
            json.dump({}, f)

    url_lastword_list = [
        re.search(r'[^a-zA-Z0-9_]([a-zA-Z0-9_]*)$', url).group()[1:]
        if re.search(r'[^a-zA-Z0-9_]([a-zA-Z0-9_]*)$', url) else url
        for url in url_list
    ]  # 按照最后一个词的频率生成词典
    url_lastword = {}  # 统计最后一个词的频率
    for i, word in enumerate(url_lastword_list):
        url_lastword[word] = url_lastword.get(word, 0) + url_num[i]
    url_lastword = sorted(url_lastword.items(),
                          key=lambda x: x[1], reverse=True)
    url_dict = [word[0]
                for word in url_lastword if word[1] >= 10]  # 筛选出出现次数多的词

    # 读取freq_dict
    freq_lastword_dict = json.load(open(freq_lastword_path, 'r'))
    freq_lastword_dict = [word.lower() for word in freq_lastword_dict]
    freq_lastword_dict = freq_lastword_dict + url_dict
    freq_lastword_dict = sorted(
        set(freq_lastword_dict), key=freq_lastword_dict.index)  # 稳定去重
    if len(freq_lastword_dict) > 10000:
        freq_lastword_dict = freq_lastword_dict[-10000:]
    json.dump(freq_lastword_dict, open(
        freq_lastword_path, 'w'), indent=4)  # 更新词典至文件

    freq_lastword_set = set(freq_lastword_dict)
    index_list = np.array(  # 以词典中的词结尾的url为安全url
        [True if word in freq_lastword_set else False for word in url_lastword_list])

    return index_list


def filter_by_special(safe_list, safe_num, url_list, url_num):
    """根据特殊字符过滤url, 这个函数不是通用的做法，根据具体情况定制
    Args:
        safe_list (list): 安全的url
        safe_num (list): 安全的url的出现次数
        url_list (list): 待过滤的url
        url_num (list): 待过滤的url的出现次数
    Returns:
        safe_list (list): 安全的url
        safe_num (list): 安全的url的出现次数
        url_list (list): 待过滤的url
        url_num (list): 待过滤的url的出现次数
    """
    # keep_url = [
    #     # 将"https://c.tb.cn/e1.zfrswwh"变成"https://c.tb.cn/e1"
    #     "https://c.tb.cn",
    #     # 往往是某个商品的二维码 将"https://m.tb.cn/h.UkSFhyY"变成"https://m.tb.cn/h"
    #     "https://m.tb.cn",
    # ]
    # index_list = np.array([False for url in url_list])
    # for i, url in enumerate(url_list):
    #     for special_url in keep_url:
    #         if special_url in url:
    #             index_list[i] = True
    #             break
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '.'.join(safe_url.split('.')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # keep_url = [
    #     # 'https://d.feizhu.com', # 飞猪各种各样的链接，保留
    #     # "https://haibao.m.taobao.com/html/1wa_dcwia"，类似这样的url是海报，保留, 37次在page2num中出现（不算频率）
    #     # 'https://haibao.m.taobao.com/html',
    #     # "https://survey.taobao.com/apps/zhiliao/m6acodnko"，类似这样的url是问卷，保留, 27次在page2num中出现（不算频率）
    #     # 'https://survey.taobao.com/apps/zhiliao/',
    #     # "https://market.m.taobao.com/markets/h5/ryysztj_copy", # 这个url不知道干啥的，保留
    #     # "https://market.m.taobao.com/app/trip/rx-shop-one/pages", # 这个url不知道干啥的，保留
    #     # "https://survey.alitrip.com/survey/kwvma3ju4"，这个url是问卷，保留
    #     # "https://survey.alitrip.com/survey/",
    #     # "https://v.ubox.cn/qr/c0820141_309_1",这个url是每一件具体的商品，保留
    #     # "https://v.ubox.cn/qr/",
    #     # "https://isite.baidu.com/site/wjzkgwn0/c5b49b38-24fd-4101-bb30-c1470444ba93", 这个url是百度商品广告，保留
    #     # "https://isite.baidu.com/site/",
    #     # https://pages.tmall.com/wow/an/tmall/default-rax/1781b61b73d，这个是天猫的啥活动，保留
    #     # "https://pages.tmall.com/wow/an/tmall/default-rax/",
    #     # https://d2cxkq4yy.wasee.com/wt/d2cxkq4yy, 这个是什么景区网站，保留
    #     # "wasee.com/wt/",
    #     # "https://go.smzdm.com/e7e98445c2688df1/ca_aa_yh_5337_69667390_14763_0_5329_0", 这个是买什么东西，保留
    #     # "https://go.smzdm.com/",
    # ]
    # index_list = np.array([False for url in url_list])
    # for i, url in enumerate(url_list):
    #     for special_url in keep_url:
    #         if special_url in url:
    #             index_list[i] = True
    #             break
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    return safe_list, safe_num, url_list, url_num


def process_by_force(url):
    """暴力处理一下url, 按照/分割去掉最后一个元素
    """
    ret_url = '/'.join(url.split('/')[:-1])
    return ret_url
