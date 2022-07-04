import numpy as np
from src import simulation, lci_test
from networkx import to_numpy_array, binomial_graph
from multiprocessing import Pool, cpu_count
import pickle
import os
from tqdm import tqdm

# Set parameters
iterations = 2
dims = 7
run_time = 200
phi_inv = lambda x: np.exp(np.minimum(x, 1) - 1)*(x<1) + x*(x>=1)
dphi_inv = lambda x: np.exp(np.minimum(x, 1)-1)*(x<1) + 1*(x>=1)
max_points = 5000
decays = 0.8 * np.ones((dims, dims))
baseline = 0.25 * np.ones(dims)
df1 = 8
df2 = 3
kappa = 1e3
kernel_support = 10

# Initialize LCI objects
LCI_1 = lci_test.LIPP(second_order=False, initialize_blocks=True, kappa=kappa, phi_inv=phi_inv, dphi_inv=dphi_inv, df1=df1, df2=df2, kernel_support=kernel_support)
LCI_2 = lci_test.LIPP(second_order=True, initialize_blocks=True, kappa=kappa, phi_inv=phi_inv, dphi_inv=dphi_inv, df1=df1, df2=df2, kernel_support=kernel_support)


def experiment_eca(n):
    np.random.seed(n)

    # Prepare experiment
    adj_true = 0.4*to_numpy_array(binomial_graph(dims, 0.2, directed=True)).T
    adj_true = adj_true * (2*np.random.binomial(n=1, p=0.5, size=adj_true.shape)-1)
    adj_true[np.diag_indices(dims)] = 0.3

    ts = simulation.get_hawkes(dims=dims, adjacency=adj_true, decays=decays, baseline=baseline,
                                kernel_support=kernel_support, max_points=max_points, run_time=run_time, phi_inv=phi_inv)
    
    LCI_1.set_data(ts, run_time=run_time)
    LCI_2.set_data(ts, run_time=run_time)
    
    adj1 = LCI_1.eca(verbose=False)
    adj2 = LCI_2.eca(verbose=False)

    return adj1, adj2, adj_true

if __name__ == "__main__":
    results = list(tqdm(Pool(cpu_count()-1).imap_unordered(experiment_eca, range(iterations)), total=iterations))

    # Check if folder exists and save results in pickle file
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/eca_results.pkl', 'wb') as f:
        pickle.dump(results, f)