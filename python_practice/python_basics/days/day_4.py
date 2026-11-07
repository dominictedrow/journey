a = "30"
b = "Days"
c = "Of"
d = "Python"
e = a + " " + b + " " + c + " " + d # Make sure to add spaces in between the strings

f = "Coding"
g = "For"
h = "All"
i = f+" "+g+" "+h 
company = "Coding For All"
#print(company)
#print(len(company))
#print(company.capitalize())
#print(company.upper())
#print(company.lower())
#print(company.swapcase())
print(company.replace("coding", "Python").replace("All", "Everyone")) #Can't make multiple changes in one replace but can replace multiple times.
print(company.split(" "))

j = "Facebook, Google, Microsoft, Apple, IBM, Oracle, Amazon"
print(company[10])
print(company.index("F"))
print(company.rfind("i"))
sentence = "You cannot end a sentence with because because because is a conjunction"
print(sentence.index("because"))
print(sentence.split("because because because"))

en = "   Coding For All      "
print(en.strip()) # Removes leading and trailing whitespace
list1 = ['Django', 'Flask', 'Bottle', 'Pyramid', 'Falcon']
print(" ".join(list1))#Need to specify injected item first I think

print("I am enjoying this challenge.\nI just wonder what is next.")
print("Name\tAge\tCountry\tCity")
print("Asabeneh\t250\tFinland\tHelsinki")

raidius = 10
pi = 3.14
square = 2
print("raidius = {}".format(raidius))
print("area = {} * {} ** {}".format(pi, raidius, square))
print("The area of a circle with radius 10 is {} meters square".format(pi * raidius ** square))

eight = 8
six = 6
print("{} + {} = {}".format(eight, six, eight + six))
print("{} - {} = {}".format(eight, six, eight - six))
print("{} * {} = {}".format(eight, six, eight * six))
print("{} / {} = {}".format(eight, six, eight / six))
print("{} % {} = {}".format(eight, six, eight % six))
print("{} // {} = {}".format(eight, six, eight // six))
print("{} ** {} = {}".format(eight, six, eight ** six))
