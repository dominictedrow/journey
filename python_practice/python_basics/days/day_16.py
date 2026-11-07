from datetime import datetime
from datetime import date
from datetime import time


present = datetime.now()
print(present)

present = present.strftime("%m/%d/%Y, %H:%M:%S") #Needs qutoes and is added by dot notation with format in ()
print(present)

today = "5 December, 2019"
today = datetime.strptime(today, "%d %B, %Y")#Variable is passed in through args not dot notation, need datetime. for comamnd to work. Also needs to be in strings format explictly so the strftime formats won't work
print(today)

newyear = date(year = 2026, month = 12, day =31)
now = date(year = 2026, month = 9, day = 10)
difference = newyear - now
print(difference)

right_now = date(year = 2026, month = 9, day = 10)
june = date(year = 1970, month = 1, day = 1)
dif = right_now-june
print(dif)