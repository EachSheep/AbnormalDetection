import json
import re
import copy

lastword_dict = json.load(open('lastword_dict.json', 'r'))
target_dict = copy.deepcopy(lastword_dict)
for value in lastword_dict:
    cur = re.split(r'[^A-Za-z0-9]', value)
    target_dict.extend(cur)
    cur = re.split(r'[^A-Za-z0-9_]', value)
    target_dict.extend(cur)
sorted_dict = sorted(set(target_dict))
json.dump(sorted_dict, open('lastword_dict.json', 'w'), indent=4)