age = input("What is your age? ")
wait = 18 - int(age)
if int(age) > 18:
    print("You are old enough to learn to drive.")
elif int(age) == 18:
    print("You just turned 18! You can learn to drive now.")
else:
    print("You need to wait " + str(wait) + " years to learn to drive.")

my_age = 37
your_age = int(input("What is your age? "))
diff = my_age - your_age
if diff == 0:
    print("We are the same age.")
elif diff > 0:
    print("I am " + str(diff) + " years older than you.")
else:
    print("You are " + str(-diff) + " years older than me.")

num1 = int(input("Enter number one: "))
num2 = int(input("Enter number two: "))
if num1 > num2:
    print("The First number is greater than the second number: " + str(num1) + " > " + str(num2))
elif num1 < num2:
    print("The First number is less than the second number: " + str(num1) + " < " + str(num2))
else:
    print("The First number is equal to the second number: " + str(num1) + " = " + str(num2))

grade = int(input("Enter your grade: "))
if grade >= 90 and grade <= 100:
    print("You got an A.")
elif grade <= 89 and grade >= 80:
    print("You got a B.")
elif grade <= 79 and grade >= 70:
    print("You got a C.")
elif grade <= 69 and grade >= 60:  
    print("You got a D.")
else:
    print("You got an F.")

season = input("Enter a season: ")
Autumn = ["September", "October", "November"]
Winter = ["December", "January", "February"]
Spring = ["March", "April", "May"]
Summer = ["June", "July", "August"]
if season in Autumn:
    print("It is Autumn.")
elif season in Winter:
    print("It is Winter.")
elif season in Spring:
    print("It is Spring.")
elif season in Summer:
    print("It is Summer.")
else:
    print("Invalid month entered.")

fruits = ['banana', 'orange', 'mango', 'lemon']
fruit = input("Enter a fruit: ")
if fruit not in fruits:
    fruits.append(fruit)
    print("You added a new fruit! Here's the list: " + fruits)
else:
    print("The fruit is already in the list. Here's the list: ")

person={
    'first_name': 'Asabeneh',
    'last_name': 'Yetayeh',
    'age': 250,
    'country': 'Finland',
    'is_married': True,
    'skills': ['JavaScript', 'React', 'Node', 'MongoDB', 'Python'],
    'address': {
        'street': 'Space street',
        'zipcode': '02210'
    }
    }

if 'skills' in person:
    print(person["skills"][len(person["skills"])//2])
    if 'Python' in person['skills']:
        print("True")
    else: 
        print("False")
    if "JavaScript" in person['skills'] and 'React' in person['skills'] and not "Python" in person['skills']:
        print("He is a front end developer.")
    elif "Node" in person['skills'] and "Python" in person['skills'] and not "MongoDB" in person['skills']:
        print("He is a back end developer.")
    elif 'React' in person['skills'] and 'Node' in person['skills'] and 'MongoDB' in person['skills'] and 'Python' in person['skills']:
        print("He is a full stack developer.")
    else:
        print("unknown title")
else:
    print("This person has no skills.")

if person['is_married'] and person['country'] == 'Finland':
    print(person['first_name'] + " " + person['last_name'] + " lives in " + person['country'] + ". He is married.")
else:
    print(person['first_name'] + " " + person['last_name'] + " lives in " + person['country'] + ". He is not married.")


