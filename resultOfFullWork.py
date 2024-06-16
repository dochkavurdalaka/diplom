from collections import defaultdict
from operator import itemgetter
import time
from docplex.mp.model import Model
from dwave.system import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer
from qiskit_optimization.translators import from_docplex_mp
from dwave.samplers import SimulatedAnnealingSampler
import networkx
from networkx.generators.random_graphs import erdos_renyi_graph
from matplotlib import pyplot as plt
import networkx as nx
from dimod import BinaryQuadraticModel, ExactDQMSolver, ExactSolver

import qubovert as qv

import dimod
import random
from qubovert.utils import solve_qubo_bruteforce
import QAOA
import pandas as pd

n = 90
p = 0.5
G = erdos_renyi_graph(n, p, seed = 1)
random.seed(1)
for i, j in G.edges():
    G[i][j]['weight'] = random.randint(1, 10)

# graph = nx.Graph()
# graph.add_nodes_from([0, 1, 2, 3, 4])
# graph.add_weighted_edges_from([(0, 1, 2), (1, 2, 3), (2, 3, 0.5), (3, 0, 1), (2, 0, 3), (4, 3, 0.5), (4, 1, 1)])

def QUBO_bruteforce(graph: networkx.Graph):
    Q = defaultdict(int)
    for i, j, w in graph.edges.data("weight", default=1):
        Q[(i, i)] += -w
        Q[(j, j)] += -w
        Q[(i, j)] += 2 * w


    bqm = dimod.BQM.from_qubo(Q)
    sampler = ExactSolver()
    response = sampler.sample(bqm)
    #print(response)
    result_list = [(sample, E) for sample, E in response.data(fields=['sample', 'energy'])]
    optimal = min(result_list, key=itemgetter(1))
    optimal_list = [sample for sample, E in result_list if E == optimal[1]]

    answer = []
    for i in optimal_list:
        answer.append(''.join(str(v[1]) for v in i.items()))

    return answer

def QUBO_simulatedannealing(graph: networkx.Graph):
    Q = defaultdict(int)
    for i, j, w in graph.edges.data("weight", default=1):
        Q[(i, i)] += -w
        Q[(j, j)] += -w
        Q[(i, j)] += 2 * w


    bqm = dimod.BQM.from_qubo(Q)
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads = 1)
    #print(response)
    result_list = [(sample, E) for sample, E in response.data(fields=['sample', 'energy'])]
    optimal = min(result_list, key=itemgetter(1))
    optimal_list = [sample for sample, E in result_list if E == optimal[1]]

    answer = []
    for i in optimal_list:
        answer.append(''.join(str(v[1]) for v in i.items()))

    return answer

def QUBO_quantumannealing(graph: networkx.Graph):
    Q = defaultdict(int)
    for i, j, w in graph.edges.data("weight", default=1):
        Q[(i, i)] += -w
        Q[(j, j)] += -w
        Q[(i, j)] += 2 * w


    bqm = dimod.BQM.from_qubo(Q)
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample(bqm, num_reads = 1)
    #print(response)
    result_list = [(sample, E) for sample, E in response.data(fields=['sample', 'energy'])]
    optimal = min(result_list, key=itemgetter(1))
    optimal_list = [sample for sample, E in result_list if E == optimal[1]]

    answer = []
    for i in optimal_list:
        answer.append(''.join(str(v[1]) for v in i.items()))

    return answer

def GoemansWillyamson(graph: networkx.Graph):
    model = Model()

    # Create n binary variables
    x = model.binary_var_list(n)

    # Define the objective function to be maximized
    model.maximize(model.sum(w * x[i] * (1 - x[j]) + w * (1 - x[i]) * x[j] for i, j, w in graph.edges.data("weight", default=1)))


    # Convert the Docplex model into a `QuadraticProgram` object
    problem = from_docplex_mp(model)
    goewill = GoemansWilliamsonOptimizer(1)
    result = goewill.solve(problem)
    #print("".join(str(i) for i in result.x))
    return "".join(str(i) for i in result.x)


graph = G

start_time = time.time()
#qubo_result = QUBO_quantumannealing(graph)

#GoemWill_result = GoemansWillyamson(graph)
#qaoa_result = QAOA.runQAOA(graph, 1)
end_time = time.time()
print(QAOA.maxcut_obj(qubo_result[0], graph))
print(end_time - start_time)
#GoemWill_result = GoemansWillyamson(graph)
#qaoa_result = QAOA.runQAOA(graph,14)

#print(qubo_result)
#print(GoemWill_result)
#print(qaoa_result)

#print(QAOA.maxcut_obj(qubo_result[0], graph), QAOA.maxcut_obj('0001111', graph))
#QAOA.runQAOA(graph,20)
#print(QAOA.maxcut_obj('1110000', graph), QAOA.maxcut_obj('0101001', graph))