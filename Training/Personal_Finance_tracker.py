import csv
income = 0
expense = 0
net = 0

with open('transactions.csv', newline='') as csvfile:
  reader = csv.DictReader(csvfile)

  for row in reader:
      
    try:
      clean = row['type'].lower().replace(' ', '')
    except KeyError:
        print('this is invalid')
        continue
    try:
      value_clean = float(row['amount'])
    except ValueError:
      print('thats not a monetary value!. Fix this value: "', row['amount'], '" For this date: "', row['date'], '"' )  
      continue
             
    if clean == 'income':
        income += value_clean
    elif clean == 'expense':
        expense += value_clean
    else:
      print("this value is ", row['type'])
      print("Value has to be either: income OR expense")

net = income - expense

print("income: ", income)
print("expenses: ", expense)
print("net cashflow: ", net)
