# sets
it_companies = {'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon'}
A = {19, 22, 24, 20, 25, 26}
B = {19, 22, 20, 25, 26, 24, 28, 27}
age = [22, 19, 24, 25, 26, 24, 25, 24]

print(len(it_companies))

it_companies.add("Twitter")
print(it_companies)

it_companies.update(["LinkedIn", "Meta", "OpenAI"])
print(it_companies)
it_companies.remove("Facebook")

C = A.union(B)
print(C)

print(A.intersection(B))
print(A.difference(B))
print(B.issubset(A))
print(A.isdisjoint(B))
print(A.symmetric_difference(B))
del A
del B

ages = set(age)
print(str(len(ages)) + " " + str(len(age)))

teachers = "I am a teacher and I love to inspire and teach people"
words = teachers.split()
print(words)
print(len(set(words)))
print(len(words))


