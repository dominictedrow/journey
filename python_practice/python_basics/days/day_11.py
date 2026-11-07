def add_numbers(one, two):
    return one + two

def area_of_circle(raidus):
    return 3.14 * raidus **2

def all_nums (*nums):
    return sum(nums)

def convert_celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

def check_season(month):
    Winter = ['December', 'January', 'February', 12, 1, 2]
    Spring = ['March', 'April', 'May', 3, 4, 5]
    Summer = ['June', 'July', 'August', 6, 7, 8]
    Autumn = ['September', 'October', 'November', 9, 10, 11]
    if month in Winter:
        return 'Winter'
    elif month in Spring:
        return 'Spring'
    elif month in Summer:
        return 'Summer'
    elif month in Autumn:
        return 'Autumn'
    else:
        return 'Invalid month'

def calculate_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)

def solve_quadratic(a, b, c, x):
    return a *x**2 + b*x + c

def print_list(list1): #make sure to iterate over the list and then print each item.
    for item in list1:
        print(item)

def reverse_list(list1):
    for item in list1[::-1]:
        print(item)

print(reverse_list([1, 2, 3, 4, 5])) #print none at the end idk why.
print(reverse_list(["A", "B", "C"]))

def capitalize_list_items(list1): #To keep this function from printing none
    for item in list1:
        print(item.upper())

print(capitalize_list_items(["hello", "world", "python"]))

def add_items(list1, items):
    for i in items:
        list1.append(i)
    return list1

print(add_items([1, 2, 3], [4, 5, 6]))

def remove_items(list1, items):
    for i in items:
        list1.remove(i)
    return list1

print(remove_items([1, 2, 3], [1, 2]))

def sum_of_numbers(num):
    return sum(range(num + 1)) #need to use +1 so that it includes the number itself in the sum.

print(sum_of_numbers(10))

def sum_of_odds(num):
    odd_count = 0
    for i in range(num + 1):
        if i % 2 != 0:
            odd_count += 1
    return odd_count

def sum_of_evens(num):
    even_count = 0
    for i in range(num +1):
        if i % 2 == 0:
            even_count += 1
    return even_count


def even_and_odds(num):
    print(sum_of_odds(num))
    print(sum_of_evens(num))

even_and_odds(100)

def factoral(num):
    factors = []
    for i in range(num + 1):
        factors.append(i)
    result = 1
    for i in factors[1:]: #Can't inlcude 0
        result *= i
    return result


print(factoral(5))

def is_empty(list1):
    if len(list1) == 0:
        print("Parameter is empty")
    else:
        print("Parameter has something")

def calculate_mean(nums):
    a = 0
    for i in nums:
        a+= i
    m = a / len(nums)
    return m
print(calculate_mean([1,2,3,4,5,6]))
def calculate_median(nums):
    nums.sort()
    print(nums[len(nums)//2])
calculate_median([1,2,3,4,5,6])
def calculate_range(nums):
    nums.sort
    a = nums[-1]
    b = nums[0]
    r = a - b
    print(r)
calculate_range([1,2,3,4,5,6])

def greet(name):
    if len(name) != 0:
        print("Hello " + name + "!")
    else:
        print("Hello Guest!")
greet("")

def show_args(**args):
    print("Recieved: ")
    for k, v in args.items():
        print("key:", k, "value:", v)
show_args(name="Alice", age=30, city="New York")

def is_prime(num): #Can't also trigger on num itself
    factors = []
    for i in range(num):
        factors.append(i)
    divided_count = -1
    for x in factors[2:]:
        if num%x == 0:
            divided_count += 1  
    prime = False
    if divided_count == -1:
        prime = True
    else:
        prime = False
    return prime
print(is_prime(13))
print(is_prime(9))

def is_unique(lst):
    set(lst)
    if len(set(lst)) == len(lst):
        print("All items are unique")
    else:
        print("Items aren't unqiue")

is_unique(['Potato', 'Tomato', 'Mango', 'Milk',])

def dtype_check(lst):
    dtype_lst= []
    for i in lst:
        dtype_lst.append(type(i))
    if len(set(dtype_lst)) == 1:
        print("All the same dtype")
    else: 
        print("Not all the samem dtype")

dtype_check(['Potato', 'Tomato', 'Mango', 'Milk',])













