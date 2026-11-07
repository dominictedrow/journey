numbers = [-4, -3, -2, -1, 0, 2, 4, 6]

evens = [i for i in numbers if i >= 0]
print(evens)

list_of_lists =[[1, 2, 3], [4, 5, 6], [7, 8, 9]]

array = [i for x in list_of_lists for i in x]
print(array)

nums = [(i, i**0, i*1, i*i, i**3, i**4, i**5, i**6, i**7) for i in range(11)]

#print(nums)

countries = [[('Finland', 'Helsinki')], [('Sweden', 'Stockholm')], [('Norway', 'Oslo')]]
new_countries = [ #When for loops get longer you can make the [] seperate to make things more clear
    [country.upper(), country[:3].upper(), city.upper()] #This repackages the values found below by capitializing the group values found
    for group in countries # this groups them together while unpacking them in a way
    for country, city in group # this then unpacks the tuple in each group
]
print(new_countries)

#Had to do a similar thing in this one where you get the pair and then just unpack the tuple afterwards and then use those variables in the dict
country_dictionary = [{"country": country.upper(), 'city': city.upper()} for group in countries for country, city in group]
print(country_dictionary)

#Seems to be a good formula, unpack tuple and then state how those unpackings are represented in the actual list at the start
names = [[('Asabeneh', 'Yetayeh')], [('David', 'Smith')], [('Donald', 'Trump')], [('Bill', 'Gates')]]
cat_names = [first + ' ' + last for name in names for first, last in name]

print(cat_names)

#lambda functions are fucking lit I love these they make so much sense.
slope = lambda x, y, x2, y2: (y-y2)/(x-x2)
print(slope(2,2,3,4)) 
















