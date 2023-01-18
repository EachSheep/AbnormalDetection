from itertools import zip_longest

res = list(zip_longest(['abc', '12', 'xyzd'], ['def', '34', 'efgh'], fillvalue=''))
print(res)