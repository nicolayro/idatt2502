import numpy as np
import matplotlib.pyplot as plt

START_POP = 10 # Beginning population size
MUT = 0.2 # Chance for mutation
TERMINATION_STATE = 0

# Helper function converting from array of bits into decimal
def _to_decimal(binary):
    decimal = 0
    for bit in binary:
        decimal = (decimal << 1) | bit
    return decimal

# Fitness function as the task describes 
def fitness(x, target):
    # Convert to binary
    x = _to_decimal(x)
    return -np.abs(x - target)


# Combine two individuals.
# The tactic used is to pick the first half from one individual,
# and the second half from the other
def combine(i1, i2):
    # Combine parents
    half = len(i1) // 2 # Integer division 
    child = np.concatenate((i1[:half], i2[half:]))
  
    # Add a random mutation
    for i in range(len(child)):
        if child[i] > 1 or child[i] < 0:
            print(child)
        # TODO: Consider non-deprecated randomness function
        rand = np.random.uniform(0, 1)
        if rand < MUT / 2:
            child[i] = 0
        elif rand > 1.0 - (MUT / 2):
            child[i] = 1

    return child

# Selection of population
def select(population, target):
    # Sort populatin based on fitness 
    scores = np.array([fitness(n, target) for n in population])
    sorted_population_indexes = scores.argsort() # Sort scores
    # print(f"population bf sort: {population[4]}")
    sort_pop = population[sorted_population_indexes[::-1]] # Sort population from scores
    # print(f"population after sort: {population[4]}")

    # growth = np.random.randint(-3, 7) # Add some varition to pop size
    new_population = np.zeros_like(population)
    for i in range(2, len(population), 2):
        new_population[i - 1] = combine(sort_pop[i - 2], sort_pop[i])
        new_population[i] = combine(sort_pop[i - 1], sort_pop[i])

    # print(f"new population: {new_population[4]}")
    return new_population 


# Initialize a popluation of bit arrays with the given sizes
def init_population(size_p, size_i):
    return np.random.randint(0, 2, (size_p, size_i))

# Fun with numbers!
def print_stats(generation, population, num_bits, target):
    print(generation, end="\t")
    print(len(population), end="\t")
    print(num_bits, end="\t")
    print(target, end="\t")
    scores = np.array([fitness(n, target) for n in population])
    print(np.amax(scores), end="\t")
    print(np.average(scores))


def run(num_bits):
    # Initialize population
    population = init_population(START_POP, num_bits)
    # Pick random number between 0 and the max number (2 ** bits)
    target = np.random.randint(0, 2 ** num_bits)
    generation = 0
    while True:
        # Check if there exists a individual with fitness 0 (termination state)
        if [x for x in population if fitness(x, target) == TERMINATION_STATE]:
            break
    
        population = select(population, target)
        generation += 1
    print_stats(generation, population, num_bits, target)
    return generation


# Run of code
bit_sizes = np.arange(4, 17)

results = {}
print("gen\tsize\tbits\ttarget\tmax\tavg")
for num_bits in bit_sizes:
    # Do three runs
    total_generations = 0
    for _ in range(3):
        final_generation = run(num_bits)
        total_generations += final_generation
    results[num_bits] = total_generations // 3

print("\nResults:")
for key in results:
    print(f"{key}: {results[key]}")

x = list(results.keys())
y = list(results.values())

plt.plot(x, y)
plt.title("Number of generations before termination")
plt.show()
