dog = {
    "name" : "Buddy",
    "breed" : "Golden Retriever",
    "age" : 3,
    "color" : "Brown",
    "legs" : 4   
}

students = {
    "first_name" : "Jon",
    "last_name" : 'Jon',
    "age" : 9,
    "grade" : 9,
    "is_martial" : False,
    "skills": ["reading", "writing", "surfing"],
    "country": "USA",
    "city": "San Diego",
    "address": "470 Camino Del Rey" 
}
print(len(students))
print(type(students["skills"]))
students["skills"].insert(3,"coding")
print(students["skills"])
keys = [students.keys()] #Keys need to be store in a value I guess
print(keys)
list1 = [students.values]
tuples = students.items()
print(tuples)

students.pop("is_martial")
print(students)
del dog