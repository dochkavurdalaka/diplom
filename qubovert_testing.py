from qubovert import QUBO
from qubovert.sim import anneal_qubo
from qubovert.utils import solve_qubo_bruteforce

# Define your QUBO matrix as a dictionary
# The keys are tuples representing the quadratic terms (i, j)
# and the values are the coefficients of these terms.
# For linear terms, use the same index (i, i).
# For example: {(0, 0): 1, (1, 1): -2, (0, 1): -1.5} represents the matrix
# [ 1  -1.5 ]
# [ 0   -2  ]

qubo_problem = {
    (0, 0): -1,
    (1, 1): 2,
    (0, 1): 1.5,
}

# Convert the dictionary to a QUBO object
qubo = QUBO(qubo_problem)

# Use the annealing simulator to solve the QUBO problem
# Here, we use `anneal_qubo` which simulates quantum annealing
qubo_energy, qubo_solution = solve_qubo_bruteforce(qubo)

# The solution is a dictionary where keys are variable indices,
# and values are the binary values (0 or 1) that minimize the QUBO function.
print("Solution to the QUBO problem:", qubo_solution)

# To evaluate the solution, use the `qubo.value(solution)` method
# This returns the value of the objective function for the given solution
#print("Objective function value:", qubo.value(solution))