height = input("Enter the height of the triangle: ")
base = input("Enter the base of the triangle: ")
area = 0.5 * float(height) * float(base)
print("This is the area of the triangle: " + str(area))

a = input("Enter the first side of the rectangle: ")
b = input("Enter the second side of the rectangle: ")
p = float(a) + float(a) + float(b) + float(b)
ar = float(a) * float(b)
print("This is the area: " + str(ar) + " and this is the perimiter: " + str(p))


r = input("Enter the radius of the circle: ")
ca = 3.14 * (float(r)**2)
c = 2 * 3.14 * float(r)
print("This is the area of the circle: " + str(ca) + " and this is the circumference: " + str(c))

x_int = 0+2/2
y_int = -2
slope1 = (y_int - 0) / (0 - x_int)
slope2 = (2 - 10)/(2 -6)
def comp(slope1, slope2):
    if slope1 == slope2:
        print("The slopes are equal")
    elif slope1 > slope2:
        print("The first slope is greater than the second slope")
    else:
        print("The second slope is greater than the first slope")
comp(slope1, slope2)

x = input("Enter a value for x: ")
x = float(x)
y = (x**2 + 6*x + 9)
print("The value of y is: " + str(y))

py = len("python")
dr = len("dragon")
print(py > dr)
print(py is dr)
print("jargon" in "I hope this course is not full of jargon")

hours = input("Enter hours: ")
rate = input("Enter rate per hour: ")
pay = float(hours) * float(rate)
print("Your weekly earning is: " + str(pay))

string = "10"
num = 10
print(type(string) == type(10)) #Use type() instead of dtype() for getting dtypes
print(int(float("9.8")) == 10) #Convert the string "9.8" to a float first, then to an integer, and compare it to 10

years = input("Enter number of years you have lived: ")
seconds = float(years) * 365 * 24 * 60 * 60
print("You have lived for " + str(seconds) + " seconds.")