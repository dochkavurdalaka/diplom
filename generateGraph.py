import networkx
import random
from networkx.generators.random_graphs import erdos_renyi_graph
n = 5
p = 0.5
G = erdos_renyi_graph(n, p, seed = 1)
random.seed(1)
for i, j in G.edges():
    G[i][j]['weight'] = random.randint(1, 10)

result = "{"
for i, j, weight in G.edges.data("weight", default=1):
    result += (f"\u007b{i}, {j}, {weight}\u007d, ")
result += f"\u007d, {n},"
print(result)