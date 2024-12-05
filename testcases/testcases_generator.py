import random

FILENAME = "n5.txt"
KNAPSACK_CAPACITY = 120
ITEMS_NUM = 5
MAX_WEIGHT = 50
MAX_VALUE = 100

def generate_knapsack_testcase(filename=FILENAME, max_capacity=KNAPSACK_CAPACITY, max_items=ITEMS_NUM, max_weight=MAX_WEIGHT, max_value=MAX_VALUE):
    capacity = max_capacity
    num_items = max_items

    items = []
    for _ in range(num_items):
        weight = random.randint(10, max_weight)
        value = random.randint(10, max_value)
        items.append((weight, value))

    with open(filename, 'w') as f:
        f.write(f"{capacity}\n")  
        f.write(f"{num_items}\n")  
        for weight, value in items:
            f.write(f"{weight} {value}\n") 

    print(f"Testcase generated and saved to {filename}")

generate_knapsack_testcase()
