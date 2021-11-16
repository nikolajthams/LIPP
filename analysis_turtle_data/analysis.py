import numpy as np
from scipy.stats import binom
import networkx as nx
import matplotlib.pyplot as plt
import pickle

with open("res1.pkl", 'rb') as input:
    results1 = pickle.load(input)
with open("res2.pkl", 'rb') as input:
    results2 = pickle.load(input)

# Plot all resulting graps
for id, (g1, g2) in enumerate(zip(results1, results2)):
    fig, ax = plt.subplots(1,2)
    fig.suptitle(f"Experiment {id+1}")

    nx.draw_circular(nx.DiGraph(g1.T), with_labels=True, ax=ax[0], connectionstyle='arc3, rad=0.1')
    nx.draw_circular(nx.DiGraph(g2.T), with_labels=True, ax=ax[1], connectionstyle='arc3, rad=0.1')
    plt.show()

# Compute frequencies
dims = results1[0].shape[0]
freq1 = np.sum(results1, axis=0)[~np.eye(dims, dtype=bool)]
freq2 = np.sum(results2, axis=0)[~np.eye(dims, dtype=bool)]

# Proportion consistent in 4 out of 5 experiments
print(2*binom.cdf(k=1, n=5, p=0.5))
print(np.isin(freq1, [0, 1, 4, 5]).mean())
print(np.isin(freq2, [0, 1, 4, 5]).mean())

# Proportion consistent in all 5 experiments
print(2*binom.cdf(k=0, n=5, p=0.5))
print(np.isin(freq1, [0, 5]).mean())
print(np.isin(freq2, [0, 5]).mean())

# Number of edges
print(freq1.mean()/5)
print(freq2.mean()/5)
