import networkx as nx
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import qiskit.quantum_info as qi




graph = nx.Graph()
graph.add_nodes_from([0, 1, 2, 3])
graph.add_weighted_edges_from([(0, 1, 2), (1, 2, 3),  (3, 0, 1), (2, 0, 3)])


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


    return qc

qc = create_qaoa_circ(graph, [1.0, 1.0] * 2)
stv1 = qi.Statevector.from_instruction(qc)
print(stv1)

