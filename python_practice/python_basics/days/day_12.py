import string
from random import randint, random

def random_user_id():
    user_id = []
    letters = string.ascii_letters
    num = input("How many IDs ")
    run = input("How long should the ID be? ")
    for x in range(0, int(num)):
        for i in range(0, int(run)):
            user_id.insert(randint(0,len(user_id)), letters[randint(0, len(letters) - 1)])
        for i in range(0, int(run)):
            user_id.insert(randint(0,len(user_id)), str(randint(0, 9)))
        print(''.join(user_id))
        user_id.clear()

random_user_id()

def rand_rgb():
    rgb =[]
    for i in range(3):
        rgb.append(randint(0, 255))
    print(rgb)

rand_rgb()

def hexa_color(num):
    color = ["#",]
    letters = string.ascii_letters
    for x in range(num):
        color = ["#",]
        for i in range(3):
            color.insert(randint(1,len(color)), letters[randint(0, 7)])
        for i in range(3):
            color.insert(randint(1,len(color)), str(randint(0, 9)))
        print(''.join(color))
        color.clear()

hexa_color(8)

def rgb_array(num):
    rgb_final =[]
    for x in range(num):
        rgb = []
        for i in range(3):
            rgb.append(randint(0, 255))
        rgb_final.append(list(rgb))
    print(rgb_final)

rgb_array(8)

def generate_colors(ct, num):
    if ct == "rgb":
        rgb_array(num)
    elif ct == "hexa": 
        hexa_color(num)

generate_colors("rgb", 3)

def shuffle_list(list1):
    shuffled_list = []
    for x in range(len(list1)):
        rand_index = randint(0,len(list1)-1)
        rand = list1[rand_index]
        shuffled_list.append(rand)
        list1.pop(rand_index)
    print(shuffled_list)

shuffle_list([1,2,3,4,5,6,7,8])

def array_make():
    array = []
    nums = [0,1,2,3,4,5,6,7,8,9]
    for x in range(7):
        array.append(nums.pop(randint(0,len(nums)-1)))
    print(array)

array_make()
