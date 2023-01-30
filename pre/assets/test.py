import json
import re

safe_dict = json.load(open('safe_dict.json', 'r'))
target_dict = []
for value in safe_dict:
    cur = re.split(r'[^A-Za-z0-9_]', value)
    target_dict.extend(cur)
sorted_dict = sorted(set(target_dict))
json.dump(sorted_dict, open('safe_dict.json', 'w'), indent=4)