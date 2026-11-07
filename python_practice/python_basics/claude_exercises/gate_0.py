from functools import reduce
import json


def char_freq(words):
    chars = list(words)
    done = []
    final = []
    for i in chars:
        if i not in done:
            count = chars.count(i)
            final.append((i,count))
            done.append(i)
    return sorted(final)[:3]

def file_time(filepath):
    with open(filepath) as f:
        file = f.read().splitlines()
    has = []
    for i in file:
        one =i.count("125674")
        if one == 1:
            has.append(i)
    return has

def apply_all(funcs, x):
    result = []
    for i in funcs:
        result.append(i(x))
    return result

squares = reduce(lambda x, y: x+y, map(lambda x: x**2,filter(lambda x: x % 2 == 0, [0,1,2,3,4,5,6,7]))) #When using filter or map just create lambda functions right then.

def deep_get(d, path, default = None):
    keys = path.split(".")
    current = d
    for i in keys: 
        if i not in current: #When a key doesn't line up the function just returns none to show that the path the user inputte was wrong
            return default
        current= current[i]
    return current