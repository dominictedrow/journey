import json
import re


def count_lines(filepath):
    with open(filepath) as f:
        file = f.read().splitlines()  
    words = [word for line in file for word in line.split()] #Needed to use double index there siimply because you index into the line and then the words in that line. Needed to use the .split method to seperate out strings
    counted = (len(file), len(words))
    return counted

'''
print(count_lines('30_days_of_Python_Git/data/obama_speech.txt'))
print(count_lines('30_days_of_Python_Git/data/donald_speech.txt'))
print(count_lines('30_days_of_Python_Git/data/michelle_obama_speech.txt'))
print(count_lines('30_days_of_Python_Git/data/melina_trump_speech.txt'))
'''

def most_spoken_lang(filepath):
    with open(filepath) as f:
        file = json.load(f)
    lang =[l for i in file for l in i["languages"]] #Need to do double variable since their is a list in the diction key value
    done = []
    final = {}
    for i in lang:
        if len(done) == 0:
            count = lang.count(i) #This does what re.search does 
            done.append(i)
            final.setdefault(i, count)
        elif i != done[-1]:
            count = lang.count(i)
            done.append(i)
            final.setdefault(i, count)
        elif i == done[-1]:
            continue
    return final

#print(most_spoken_lang('30_days_of_Python_Git/data/countries-data.py'))

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
    return sorted(final, key =lambda x: x[1], reverse =True)

#print(find_most_common_words('30_days_of_Python_Git/data/melina_trump_speech.txt', 10))

def email_finder(pathname):
    with open(pathname) as f:
        file = f.read()
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', file)
    return emails

#print(email_finder('30_days_of_Python_Git/data/email_exchanges_big.txt'))

def sim(filepath1, filepath2):
    speech1 = find_most_common_words(filepath1)
    speech2 = find_most_common_words(filepath2)
    word, count = [i for i in speech1]
    word2, count2 = [i for i in speech2]
    done = []
    dif = []
    for i in word:
        if i not in done and word[len(done)] == word2[len(done)]:
            dif.append(count[0]-count2[0])
            count.remove(0)
            count2.pop(0)
    dif_total = dif.items.sum




