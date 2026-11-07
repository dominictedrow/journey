import re 

paragraph = 'I love teaching. If you do not love teaching what else can you love. I love Python if you do not love something which can give you all the capabilities to develop an application what else can you love.'
words = re.findall(r'\w+', paragraph) #r/w+ just gets all of the words from a string

patterns = []
done = [] 
for word in words:
    if word not in done:
        count = words.count(word) #I guess there was a counting operator that found the number of things inside a list. That would've made things a lot easier
        pattern = count, word 
        patterns.append(pattern)
        done.append(word)

#print(patterns)

points = ['-12', '-4', '-3', '-1', '0', '4', '8']
point = [int(i) for i in points ]
sorted(point)
print(point)
print(point[0]-point[-1])

def is_valid(name):
    if re.search('nigga', name) == None:
        print(False)
    else:
        print(True)

is_valid('Nigga')
is_valid('nigga')
is_valid('whore')

sentence = '''%I $am@% a %tea@cher%, &and& I lo%#ve %tea@ching%;. There $is nothing; &as& mo@re rewarding as educa@ting &and& @emp%o@wering peo@ple. ;I found tea@ching m%o@re interesting tha@n any other %jo@bs. %Do@es thi%s mo@tivate yo@u to be a tea@cher!?'''

clean1 = re.sub('%', '', sentence)
clean2 = re.sub('@', '', clean1)
clean3 = re.sub('#', '', clean2)
clean4 = re.sub('&', '',clean3)
clean5= re.sub(';', '', clean4)
new = re.sub('\$', '', clean5)

print(new)

swords = re.findall(r'\w+', new)
sentence_counter = []
sdone = []
for word in swords:
    if word not in sdone:
        count = swords.count(word) #I guess there was a counting operator that found the number of things inside a list. That would've made things a lot easier
        pattern = count, word 
        sentence_counter.append(pattern)
        sdone.append(word)
print(sentence_counter)