import os
import pymongo
#When using a conda env you need to change the interpeter in the bottom right corner.

MONGO_URI = 'mongodb+srv://dominictedrow_db_user:S2j1yXtCyorTzPtd@cluster0.iuzyb2r.mongodb.net/?appName=Cluster0'
client = pymongo.MongoClient(MONGO_URI)

courses = [
    {"course": "AP Physics", "students": 24, "difficulty": 5, "department": "Science"},
    {"course": "AP Biology", "students": 30, "difficulty": 4, "department": "Science"},
    {"course": "AP Economics", "students": 27, "difficulty": 3, "department": "Social Science"},
    {"course": "AP Calculus", "students": 22, "difficulty": 5, "department": "Math"},
    {"course": "Spanish IV", "students": 18, "difficulty": 3, "department": "Language"},
    {"course": "World History", "students": 26, "difficulty": 2, "department": "Social Science"}
]

db = client['course']

db.course.insert_many(courses)

query1 = {
    "difficulty": {"gte": 4}, #Comparison operators are acronyms 
}
fquery1 = db.course.find(query1)
print(fquery1)

query2 = {
    "difficulty": {"gte": 4},
    "students": {"gte": 20} #Treats multiple query statemnts as AND
}
fquery2 = db.course.find(query2)
print(fquery2)

vars = db.course.find({}, {"course": 1, "students": 0, "difficulty": 1, "department": 0})
print(vars)

db.course.find().sort("difficulty") #Specify value being sorted. Every instance needs to have this though probably

db.course.find().limit(3)

db.course.update_one({"course": "AP Economics"}, {'$set':{'difficulty':4}})

query3 = {"students": {"lt": 20}}

db.course.delete_many(query3)

for i in db.course.find():
    print(i)

db.course.drop




