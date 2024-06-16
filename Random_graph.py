from collections import defaultdict
from operator import itemgetter

from networkx.generators.random_graphs import erdos_renyi_graph
from matplotlib import pyplot as plt
import networkx as nx
from dimod import BinaryQuadraticModel, ExactDQMSolver, ExactSolver
import dimod
import random
import pandas

n = 5
p = 0.4
G = erdos_renyi_graph(n, p, seed = 1)
random.seed(1)

for i, j in G.edges():
    #G[i][j]['weight'] = round(random.uniform(1, 5), 2)
    G[i][j]['weight'] = round(random.uniform(1, 7), 0)

# G = nx.Graph()
# G.add_nodes_from([0, 1, 2, 3, 4])
# G.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (2, 3, 0.5), (3, 0, 1.0), (2, 0, 3.0), (4, 3, 0.5), (4, 1, 1.0)])
num_nodes = G.number_of_nodes()


print(G.nodes)
print(G.edges)

f, ax = plt.subplots(1, 2)
ax[0].set_title('input')
ax[1].set_title('result')

pos = nx.spring_layout(G, seed=2)
nx.draw(G,pos, with_labels=True, alpha=0.8, style='solid', node_size=500, ax=ax[0])
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax[0])


Q = defaultdict(float)

# Update Q matrix for every edge in the graph
for i, j, w in G.edges.data("weight", default=1):
    Q[(i,i)] += -w
    Q[(j,j)] += -w
    Q[(i,j)] += 2.0 * w


bqm = dimod.BQM.from_qubo(Q)
#print(bqm.to_polystring())
sampler = ExactSolver()
response = sampler.sample(bqm)

#temp = response.to_pandas_dataframe()

#print(temp.iloc[55]['energy'] - temp.iloc[29]['energy'])
#print(response)

result_list = [(sample, E) for sample, E in response.data(fields=['sample','energy'])]

optimal = min(result_list, key=itemgetter(1))

optimal_list = [sample for sample, E in result_list if E==optimal[1]]

for i in optimal_list:
    print(''.join(str(v[1]) for v in i.items()))

lut = optimal_list[0]


S0 = [node for node in G.nodes if not lut[node]]
S1 = [node for node in G.nodes if lut[node]]
cut_edges = [(u, v) for u, v in G.edges if lut[u]!=lut[v]]
uncut_edges = [(u, v) for u, v in G.edges if lut[u]==lut[v]]


nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='r')
nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='c')
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
nx.draw_networkx_labels(G, pos)


figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
