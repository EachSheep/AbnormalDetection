import json
import re
import numpy as np
import os

def filter_by_ifurl1(safe_list, safe_num, url_list, url_num):
    """根据是否是url过滤
    Args:
        safe_list (list): 安全的url
        safe_num (list): 安全的url对应的访问次数
        url_list (list): 待过滤的url
        url_num (list): 待过滤的url对应的访问次数
    Returns:
        safe_list (list): 安全的url
        safe_num (list): 安全的url对应的访问次数
        url_list (list): 待过滤的url
        url_num (list): 待过滤的url对应的访问次数
    """

    # 非url的加入safe_list
    remove_index = []
    for i, url in enumerate(url_list):
        if not re.match(r'^https?:\/\/', url):
            safe_list.append(url)
            safe_num.append(url_num[i])
            remove_index.append(False)
        else:
            remove_index.append(True)
    url_list = np.array(url_list)[np.array(remove_index) == True].tolist()
    url_num = np.array(url_num)[np.array(remove_index) == True].tolist()
    return safe_list, safe_num, url_list, url_num


def filter_by_extension1(safe_list, safe_num, url_list, url_num):
    """根据拓展名过滤url
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
    extensions = set([extension.lower() for extension in json.load(
        open('pre/assets/extension_of_filename.json', 'r'))])  # 以各种拓展名结尾的

    index_list = np.array(
        [True if url.split('.')[-1] in extensions else False for url in url_list])
    
    safe_list.extend(np.array(url_list)[index_list].tolist())
    safe_num.extend(np.array(url_num)[index_list].tolist())
    url_list = np.array(url_list)[index_list == False].tolist()
    url_num = np.array(url_num)[index_list == False].tolist()
    return safe_list, safe_num, url_list, url_num


def filter_by_equal1(safe_list, safe_num, url_list, url_num):
    """根据=过滤url
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
    index_list = np.array([True if '=' in url else False for url in url_list])
    tmp_list = np.array(url_list)[index_list].tolist()
    tmp_num = np.array(url_num)[index_list].tolist()
    url_list = np.array(url_list)[index_list == False].tolist()
    url_num = np.array(url_num)[index_list == False].tolist()

    tmp = {}
    for i in range(len(tmp_list)):
        tmp[tmp_list[i]] = tmp.get(tmp_list[i], 0) + tmp_num[i]
    safe_list.extend(tmp.keys())
    safe_num.extend(tmp.values())

    return safe_list, safe_num, url_list, url_num


def filter_by_dict1(safe_list, safe_num, url_list, url_num):
    """根据词典过滤url，以词典中的词结尾的url为安全url
    safe_dict: 认为加入的，一定是关键词的词典
    freq_dict: 自动根据当天的频率生成的词典，最多保留10000条，超了过后，旧的词会被删除

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
    if not os.path.exists('pre/assets/'):
        os.makedirs('pre/assets/')
    if not os.path.exists('pre/assets/safe_dict.json'):
        with open('pre/assets/safe_dict.json', 'w') as f:
            json.dump({}, f)
    if not os.path.exists('pre/assets/freq_dict.json'):
        with open('pre/assets/freq_dict.json', 'w') as f:
            json.dump({}, f)

    url_lastword_list = [url.split('/')[-1]
                         for url in url_list]  # 按照最后一个词的频率生成词典
    url_lastdict = {}
    for i, word in enumerate(url_lastword_list):
        url_lastdict[word] = url_lastdict.get(word, 0) + url_num[i]
    url_lastword = sorted(url_lastdict.items(),
                          key=lambda x: x[1], reverse=True)
    # 筛选出出现次数大于等于5的词
    url_dict = [word[0] for word in url_lastword if word[1] >= 5]

    # 读取safe_dict
    safe_dict = json.load(open('pre/assets/safe_dict.json', 'r'))
    safe_dict = [word.lower() for word in safe_dict]
    safe_dict = sorted(set(safe_dict))
    json.dump(safe_dict, open('pre/assets/safe_dict.json', 'w'),
              indent=4)  # 更新词典至文件
    # 读取freq_dict
    freq_dict = json.load(open('pre/assets/freq_dict.json', 'r'))
    freq_dict = [word.lower() for word in freq_dict]
    freq_dict = sorted(set(freq_dict) - set(safe_dict),
                       key=freq_dict.index)  # 稳定去重
    url_dict = sorted(set(url_dict) - set(safe_dict),
                      key=url_dict.index)  # 稳定去重
    freq_dict = freq_dict + url_dict
    freq_dict = sorted(set(freq_dict), key=freq_dict.index)  # 稳定去重
    if len(freq_dict) > 10000:
        freq_dict = freq_dict[-10000:]
    json.dump(freq_dict, open('pre/assets/freq_dict.json', 'w'),
              indent=4)  # 更新词典至文件

    # 合并词典
    safe_dict.extend(freq_dict)
    safe_dict = list(set(safe_dict))

    safe_dict = set(safe_dict)
    # 以词典中的词结尾的url为安全url
    index_list = np.array(
        [True if word in safe_dict else False for word in url_lastword_list])
    safe_list.extend(np.array(url_list)[index_list].tolist())
    safe_num.extend(np.array(url_num)[index_list].tolist())
    url_list = np.array(url_list)[index_list == False].tolist()
    url_num = np.array(url_num)[index_list == False].tolist()
    return safe_list, safe_num, url_list, url_num

def filter_by_cnt1(safe_list, safe_num, url_list, url_num):
    """根据出现次数过滤
    Args:
        safe_list (list): 安全的url
        safe_num (list): 安全的url对应的访问次数
        url_list (list): 待过滤的url
        url_num (list): 待过滤的url对应的访问次数
    Returns:
        safe_list (list): 安全的url
        safe_num (list): 安全的url对应的访问次数
        url_list (list): 待过滤的url
        url_num (list): 待过滤的url对应的访问次数
    """
    # 出现次数大于10的加入safe_list
    remove_index = []
    for i, url in enumerate(url_list):
        if url_num[i] > 10:
            safe_list.append(url)
            safe_num.append(url_num[i])
            remove_index.append(False)
        else:
            remove_index.append(True)
    url_list = np.array(url_list)[np.array(remove_index) == True].tolist()
    url_num = np.array(url_num)[np.array(remove_index) == True].tolist()
    return safe_list, safe_num, url_list, url_num


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
    # 将这样的url "https://d.feizhu.com/xxxxxxx"变成"https://d.feizhu.com"
    special_url_list = [  # 含有这些字符的url，全部变成这些字符，并加入safe_list
        'https://d.feizhu.com',
    ]

    for special_url in special_url_list:
        index_list = np.array(
            [True if special_url in url else False for url in url_list])
        safe_list.append(special_url)
        safe_num.append(int(np.sum(np.array(url_num)[index_list == True])))
        url_list = np.array(url_list)[index_list == False].tolist()
        url_num = np.array(url_num)[index_list == False].tolist()

    # 将这样的url "https://c.tb.cn/e1.zfrswwh"变成"https://c.tb.cn/e1"
    special_url = "https://c.tb.cn/"  # 先筛出含有这些字符的url，再根据后面的字符进行筛选
    index_list = np.array(
        [True if special_url in url else False for url in url_list])
    tmp_url_list = np.array(url_list)[index_list == True].tolist()
    tmp_num_list = np.array(url_num)[index_list == True].tolist()
    url_list = np.array(url_list)[index_list == False].tolist()
    url_num = np.array(url_num)[index_list == False].tolist()
    tmp = {}
    for i, safe_url in enumerate(tmp_url_list):
        tmp_url_list[i] = special_url + re.split('\/|\.', safe_url)[-2]
        tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    safe_list.extend(tmp.keys())
    safe_num.extend(tmp.values())

    # 最后带有点的url经过观察，往往是某个商品的二维码 https://m.tb.cn/h.UkSFhyY
    index_list = np.array(
        [True if 'm.tb.cn' in url else False for url in url_list])
    tmp_url_list = np.array(url_list)[index_list == True].tolist()
    tmp_num_list = np.array(url_num)[index_list == True].tolist()
    url_list = np.array(url_list)[index_list == False].tolist()
    url_num = np.array(url_num)[index_list == False].tolist()
    tmp = {}
    for i, safe_url in enumerate(tmp_url_list):
        tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
        tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    safe_list.extend(tmp.keys())
    safe_num.extend(tmp.values())

    # 特殊处理"https://pontus-image-common.oss-cn-hangzhou.aliyuncs.com/img/2214673369151/7f2ccedf3d2a4e7586097f3bff9dfbb2/xxxxx"
    index_list = np.array(
        [True if 'pontus-image-common.oss-cn-hangzhou.aliyuncs.com' in url else False for url in url_list])
    tmp_url_list = np.array(url_list)[index_list == True].tolist()
    tmp_num_list = np.array(url_num)[index_list == True].tolist()
    url_list = np.array(url_list)[index_list == False].tolist()
    url_num = np.array(url_num)[index_list == False].tolist()
    tmp = {}
    for i, safe_url in enumerate(tmp_url_list):
        tmp_url_list[i] = '/'.join(safe_url.split('/')[:-2])
        tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    safe_list.extend(tmp.keys())
    safe_num.extend(tmp.values())

    # 特殊处理"https://d2cxkq4yy.wasee.com/wt/xxx"
    index_list = np.array(
        [True if 'wasee.com' in url else False for url in url_list])
    tmp_url_list = np.array(url_list)[index_list == True].tolist()
    tmp_num_list = np.array(url_num)[index_list == True].tolist()
    url_list = np.array(url_list)[index_list == False].tolist()
    url_num = np.array(url_num)[index_list == False].tolist()
    tmp = {}
    for i, safe_url in enumerate(tmp_url_list):
        tmp_url_list[i] = "https://wasee.com/" + \
            '/'.join(safe_url.split('/')[3:-1])
        tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    safe_list.extend(tmp.keys())
    safe_num.extend(tmp.values())

    # 粗略的暴力筛选: .com, .cn, .net等按照/分割去掉最后一个元素
    index_list = np.array([True for url in url_list])
    tmp_url_list = np.array(url_list)[index_list == True].tolist()
    tmp_num_list = np.array(url_num)[index_list == True].tolist()
    url_list = np.array(url_list)[index_list == False].tolist()
    url_num = np.array(url_num)[index_list == False].tolist()
    tmp = {}
    for i, safe_url in enumerate(tmp_url_list):
        tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
        tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    safe_list.extend(tmp.keys())
    safe_num.extend(tmp.values())

    # # "https://haibao.m.taobao.com/html/1wa_dcwia"，类似这样的url是海报，保留
    # index_list = np.array([True if 'haibao.m.taobao.com' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # "https://survey.taobao.com/apps/zhiliao/m6acodnko"，类似这样的url是问卷，保留
    # index_list = np.array([True if 'survey.taobao.com' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # "https://market.m.taobao.com/markets/h5/ryysztj_copy"
    # # "https://market.m.taobao.com/app/trip/rx-shop-one/pages/null"，这两个url不知道干啥的，保留
    # index_list = np.array([True if 'market.m.taobao.com' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # "https://survey.alitrip.com/survey/kwvma3ju4"，这个url是问卷，保留
    # index_list = np.array([True if 'survey.alitrip.com' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # "https://v.ubox.cn/qr/c0820141_309_1",这个url是每一件具体的商品，保留
    # index_list = np.array([True if 'v.ubox.cn' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # "https://isite.baidu.com/site/wjzkgwn0/c5b49b38-24fd-4101-bb30-c1470444ba93", 这个url是百度商品广告，保留
    # index_list = np.array([True if 'isite.baidu.com/site' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # https://pages.tmall.com/wow/an/tmall/default-rax/1781b61b73d，这个是天猫的啥活动，保留
    # index_list = np.array([True if 'pages.tmall.com' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # https://d2cxkq4yy.wasee.com/wt/d2cxkq4yy, 这个是什么景区网站，保留
    # index_list = np.array([True if 'wasee.com' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # "https://go.smzdm.com/e7e98445c2688df1/ca_aa_yh_5337_69667390_14763_0_5329_0", 这个是买什么东西，保留
    # index_list = np.array([True if 'go.smzdm.com' in url else False for url in url_list])
    # safe_list.extend(np.array(url_list)[index_list == True].tolist())
    # safe_num.extend(np.array(url_num)[index_list == True].tolist())
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()

    # # "https://ur.alipay.com/1s1c6d"，这个url是支付宝的，变成"https://ur.alipay.com"
    # index_list = np.array([True if 'ur.alipay.com' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # #  "https://s.click.taobao.com/x3x55ou", 这个url是淘宝的，变成"https://s.click.taobao.com"
    # index_list = np.array([True if 's.click.taobao.com' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # # "https://p.tb.cn/2lhtuv"，这个url是淘宝的，变成"https://p.tb.cn"
    # index_list = np.array([True if 'p.tb.cn' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # # "https://t.tb.cn/2xojdcmevlwrtk9pv5mhnb", 这个url是淘宝的，变成"https://t.tb.cn"
    # index_list = np.array([True if 't.tb.cn' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # # "https://l.eubrmb.com/q/2pnfclr4zu3", 这个url不知道干嘛的，变成"https://l.eubrmb.com"
    # index_list = np.array([True if 'l.eubrmb.com' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())

    # # "https://f.m.taobao.com/wow/z/pcraft/btrip/3rtdt3xhrdtnyn4erpiq", 这个url不知道干嘛的，变成"https://f.m.taobao.com/wow/z/pcraft/btrip"
    # index_list = np.array([True if 'f.m.taobao.com' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())

    # # "https://u.tb.cn/1weulynkkzzme7yuvjbcrt", 这个url不知道干嘛的，变成"https://u.tb.cn"
    # index_list = np.array([True if 'u.tb.cn' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # # "https://weixin.qq.com/r/leza2mhe2zymryw79xk_", 这个url微信链接，变成"https://weixin.qq.com/r"
    # index_list = np.array([True if 'weixin.qq.com' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # # "https://3hours.taobao.com/wow/z/3hours/default/n45tyrxrh4bas7aw3wk3"
    # # "https://3hours.taobao.com/17lai", 这个不知道干嘛的，变成"https://3hours.taobao.com"
    # index_list = np.array([True if '3hours.taobao.com' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # # "https://tb.cn/w4aqcev", 不知道干嘛的，变成"https://tb.cn"
    # index_list = np.array([True if 'tb.cn' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # # "https://pontus-image-common.oss-cn-hangzhou.aliyuncs.com/img/3236747168/2b5e6953654749068f813984b7c44b17/seller_id414581727335690310": 2,
    # # "https://pontus-image-common.oss-cn-hangzhou.aliyuncs.com/img/3236747168/d11590f38a584f85952beb1a437f5e76/seller_id414581739011034150": 2,
    # # 下载什么东西，变成"https://pontus-image-common.oss-cn-hangzhou.aliyuncs.com"
    # index_list = np.array([True if 'pontus-image-common.oss-cn-hangzhou.aliyuncs.com' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-2])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    # # "https://v.douyin.com/kdsapte"，抖音的什么视频
    # index_list = np.array([True if 'v.douyin.com' in url else False for url in url_list])
    # tmp_url_list = np.array(url_list)[index_list == True].tolist()
    # tmp_num_list = np.array(url_num)[index_list == True].tolist()
    # url_list = np.array(url_list)[index_list == False].tolist()
    # url_num = np.array(url_num)[index_list == False].tolist()
    # tmp = {}
    # for i, safe_url in enumerate(tmp_url_list):
    #     tmp_url_list[i] = '/'.join(safe_url.split('/')[:-1])
    #     tmp[tmp_url_list[i]] = tmp.get(tmp_url_list[i], 0) + tmp_num_list[i]
    # safe_list.extend(tmp.keys())
    # safe_num.extend(tmp.values())

    return safe_list, safe_num, url_list, url_num

if __name__ == "__main__":
    safe_list, safe_num, url_list, url_num = filter_by_ifurl1(safe_list, safe_num, url_list, url_num)
    safe_list, safe_num, url_list, url_num = filter_by_extension1(safe_list, safe_num, url_list, url_num) # 通用的清洗
    safe_list, safe_num, url_list, url_num = filter_by_equal1(safe_list, safe_num, url_list, url_num) # 通用的清洗
    safe_list, safe_num, url_list, url_num = filter_by_dict1(safe_list, safe_num, url_list, url_num) # 通用的清洗
    safe_list, safe_num, url_list, url_num = filter_by_cnt1(safe_list, safe_num, url_list, url_num) # 这个操作放后面，cnt清洗可能出错