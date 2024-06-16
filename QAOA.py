from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx

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
    """Computes expectation value based on measurement results
    Args:
        counts: (dict) key as bit string, val as count
        graph: networkx graph
    Returns:
        avg: float
             expectation value
    """
    avg = 0
    sum_count = 0

    

    for bit_string, count in counts.items():
        obj = maxcut_obj(bit_string, graph)
        avg += obj * count
        sum_count += count
    return avg/sum_count


# We will also bring the different circuit components that
# build the qaoa circuit under a single function
def create_qaoa_circ(graph, theta):
    """Creates a parametrized qaoa circuit
    Args:
        graph: networkx graph
        theta: (list) unitary parameters
    Returns:
        (QuantumCircuit) qiskit circuit
    """
    nqubits = len(graph.nodes())
    n_layers = len(theta)//2  # number of alternating unitaries
    beta = theta[:n_layers]
    gamma = theta[n_layers:]

    qc = QuantumCircuit(nqubits)

    # initial_state
    qc.h(range(nqubits))

    for layer_index in range(n_layers):
        # problem unitary
        for i, j, weight in graph.edges.data("weight", default=1):
            qc.rzz(2 * weight * gamma[layer_index], i, j)
        # mixer unitary
        for qubit in range(nqubits):
            qc.rx(2 * beta[layer_index], qubit)

    qc.measure_all()
    return qc


# Finally we write a function that executes the circuit
# on the chosen backend
def get_expectation(graph, shots):
    """Runs parametrized circuit
    Args:
        graph: networkx graph
    """
    backend = Aer.get_backend('qasm_simulator')

    def execute_circ(theta):
        qc = create_qaoa_circ(graph, theta)
        counts = backend.run(qc, seed_simulator=10,
                             shots=shots).result().get_counts()
        return compute_expectation(counts, graph)

    return execute_circ

def runQAOA(graph: networkx.Graph, num_layers: int):
    num_nodes = graph.number_of_nodes()
    p = 10000
    expectation = get_expectation(graph, p)
    res = minimize(expectation,
                   [1.0, 1.0] * num_layers,
                   method='COBYLA')


    backend = Aer.get_backend('aer_simulator')
    qc_res = create_qaoa_circ(graph, res.x)

    counts = backend.run(qc_res, seed_simulator=10, shots = p).result().get_counts()
    max_value = max(counts.values())

    answer = []
    for key in counts:
        if counts[key] == max_value:
            answer.append(key)

    #plot_histogram(counts, figsize=(15, 5))
    #plt.show()
    return answer