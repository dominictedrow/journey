import json
import re

def find_most_common_words(filepath):
    with open(filepath) as f:
        file = f.read()
    words = re.findall(r'\w+', file) #re things are really good at getting and counting strings so just use that instead of count
    done = []
    final = []
    for i in words:
        if i not in done: #First and last scenerio already accounted for
            count = words.count(i)
            done.append(i)
            final.append((i, count))
    return sorted(final, key =lambda x: x[1], reverse =True)[:10]

print(find_most_common_words('30_days_of_Python_Git/data/romeo_and_juliet.txt'))

'''
This day was about installing and using packages with pip I will learn the packages I need as I nee them
Most of the exercises where using the packages that they showed in the thing but I really don't need to learn those for my
puroposes
'''










