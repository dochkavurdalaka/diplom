from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx
import networkx as nx
import numpy as np

def maxcut_obj(solution, graph):
    """Given a bit string as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.
    Args:
        solution: (str) solution bit string
        graph: networkx graph
    Returns:
        obj: (float) Objective
    """
    # pylint: disable=invalid-name
    obj = 0
    for i, j, weight in graph.edges.data("weight", default=1):
        if solution[i] != solution[j]:
            obj -= weight
    return obj


def compute_expectation(counts, graph):
    avg = 0
    sum_count = 0

    for bit_string, count in counts.items():
        obj = maxcut_obj(bit_string, graph)
        avg += obj * count
        sum_count += count
    return avg


# We will also bring the different circuit components that
# build the qaoa circuit under a single function



def maxcut_obj2(sol, graph):
    """Given a bit string as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.
    Args:
        solution: (str) solution bit string
        graph: networkx graph
    Returns:
        obj: (float) Objective
    """
    # pylint: disable=invalid-name
    obj = 0
    for i, j, weight in graph.edges.data("weight", default=1):
        probability = (1 - sol[i])*sol[j] + (1 - sol[j])*sol[i]
        print('ll;',' ',sol[i],' ', sol[j], ' ',probability)
        obj -= weight * probability
    return obj





def main():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_weighted_edges_from([(0, 1, 10), (1, 2, 1), (0, 2, 0.5)])
    counts = dict()
    counts['000'] = 0.1
    counts['001'] = 0.2
    counts['010'] = 0.1
    counts['011'] = 0.15
    counts['100'] = 0.2
    counts['101'] = 0.1
    counts['110'] = 0.1
    counts['111'] = 0.05
    s = compute_expectation(counts, graph)
    print(s)

    counts2 = list()
    for i in range(0, 8):
        l = [float(j) for j in format(i, '03b')]
        counts2.append(l)
        #print(format(i, '03b'))

    temp = np.array([0.] * 3)

    for i, cifir in enumerate(counts.values()):
        print(np.array(counts2[i]) * cifir)
        temp += np.array(counts2[i]) * cifir

    print("\n")
    for i in temp:
        print(i)

    print("\n")
    print(maxcut_obj2(temp, graph))


main()