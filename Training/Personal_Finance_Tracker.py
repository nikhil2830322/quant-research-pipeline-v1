import csv 

total_spending = 0 
highest_amount = 0 
average_expense = 0 
category_totals = {} 
category_averages = {} 
category_highest = {} 
row_count = 0 


csv_file = "csv.csv" 

def load_csv(csv_file): 
    with open(csv_file, 'r') as file: 
        reader = csv.DictReader(file) 
        for row in reader: 
            print(row) 

def convert_to_float():
  for row in reader:
    row["amount"] = float(row["amount"])

load_csv(csv_file) 
#what i have for now 
