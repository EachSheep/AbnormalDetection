import re

def preprocess(url):
    """预处理
    Args:
        url (str): 待处理的url
    Returns:
        url (str): 处理后的url
    # 筛除局域网内广播信息
    # index_list = [True if not re.match(r'^https?:\/\/(192\.168|10|172\.1[6-9]|172\.2[0-9]|172\.3[0-1])\.', url) else False for url in url_list]
    """
    # 去除url中的汉字, 去除所有的url中文编码, 去除所有逗号, 去除所有~
    url = re.sub(r'[\u4e00-\u9fa5]|%[a-fA-F\d]{2}|~|,', '', url)
    # "https://m.amap.com/navigation/carmap/&saddr=121.834638%2c29.847424%2c%e6%88%91%e7%9a%84%e4%bd%8d%e7%bd%ae&daddr=121.51234007%2c29.84995423%2c%e5%ae%81%e6%b3%a2%e5%ae%a2%e8%bf%90%e4%b8%ad%e5%bf%83%e5%9c%b0%e9%93%81%e7%ab%99c%e5%8f%a3"
    # 变成：https://m.amap.com/navigation/carmap/&saddr=daddr=
    if '=' in url:
        url = re.sub(r'=.*&|=.*$', '=', url)
    
    url = re.sub(r'/+$|=+$|-+$', '', url)  # 末尾不能以/, =结尾
    return url