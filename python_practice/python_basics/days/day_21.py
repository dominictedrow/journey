class DF:
    def __init__(self, data):
        self.data = list(data)
    def median(self):
        middle = len(self.data)//2
        return sorted(self.data)[middle]
    def mean(self):
        mean = sum(self.data)/len(self.data)
        return mean
    def sum(self):
        return sum(self.data)
    def mode(self):
        done = []
        final = {}
        for i in self.data:
            if i not in done:
                done.append(i)
                counts = self.data.count(i)
                final.setdefault(i, counts)
        return max(final, key= final.get)
    def max(self):
        return sorted(self.data)[-1]
    def min(self):
        return sorted(self.data)[0]
    def count(self):
        count = 0
        for i in self.data:
            count += 1
        return count
    def range(self):
        return (sorted(self.data)[-1]) - (sorted(self.data)[0])
    def info(self):
        print(self.count())
        print(self.sum())
        print(self.min())
        print(self.max())
        print(self.range())
        print(self.mean())
        print(self.median())
        print(self.mode())

ages = DF([31, 26, 34, 37, 27, 26, 32, 32, 26, 27, 27, 24, 32, 33, 27, 25, 26, 38, 37, 31, 34, 24, 33, 29, 26])

ages.info()

class PersonAccount:
    def __init__(self, firstname, lastname, incomes, expenses):
        self.firstname = firstname
        self.lastname = lastname
        self.incomes = incomes
        self.expenses = expenses
    def totalincome(self):
        keys = [i for i in self.incomes.keys()]
        return sum(keys)
    def totalexpenses(self):
        keys = [i for i in self.expenses.keys()]
        return sum(keys)
    def add_exepense(self, expense, explained):
        self.expenses.setdefault(expense, explained)
        return self.expenses
    def add_income(self, income, explained):
            self.incomes.setdefault(income, explained)
            return self.incomes
    def account_balance(self):
        return self.totalincome - self.totalexpenses
    def account_info(self):
        print(self.firstname + self.lastname)
        print(self.totalincome)
        print(self.totalexpenses)
        print(self.account_balance)

expense_dict = {}
income_dict = {}
person = PersonAccount('Dominic', 'Tedrow', income_dict, expense_dict)
person.account_info()








