import random


KNAPSACK_CAPACITY = 8000
ITEMS_NUM = 1000
GENERATIONS = 2000
MAX_WEIGHT = 50
MAX_VALUE = 100
POP_SIZE = 8000 #base 1000
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
FILENAME = f"n{POP_SIZE}.txt"

def generate_knapsack_testcase(filename=FILENAME, max_capacity=KNAPSACK_CAPACITY, max_items=ITEMS_NUM, max_weight=MAX_WEIGHT, max_value=MAX_VALUE, generations = GENERATIONS):
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
        f.write(f"{generations}\n")
        f.write(f"{POP_SIZE}\n")
        f.write(f"{MUTATION_RATE}\n")
        f.write(f"{CROSSOVER_RATE}\n")
        for weight, value in items:
            f.write(f"{weight} {value}\n") 

    print(f"Testcase generated and saved to {filename}")

generate_knapsack_testcase()
