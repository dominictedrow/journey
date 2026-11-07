empty_tuple = ()
brothers = ("Bob", "Jameson", "Jai", "Chris")#In order to modify a tuple need to convert into a list
sisters = ("Alice", "Diana", "Elena")
siblings = brothers + sisters
num_siblings = len(siblings)
family_members = ("Dad", "Mom")
family_members = siblings + family_members
print(family_members)
brother1, brother2, brother3, brother4, sister1, sister2, sister3, mom, dad = family_members

fruit = ("Apple", "Banana", "Cherry")
veg = ("Carrot", "Broccoli", "Spinach")
meat = ("Chicken", "Beef", "Fish")
foods = fruit + veg + meat
food_list = list(foods)
middle = food_list[len(food_list) // 2]
print(food_list[len(food_list) // 2])
print(food_list[:3]) 
del foods 

nordic_countries = ('Denmark', 'Finland','Iceland', 'Norway', 'Sweden')
print("Estonia" in nordic_countries)
print("Iceland" in nordic_countries)



