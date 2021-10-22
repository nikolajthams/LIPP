import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from multiprocessing import cpu_count
from src.simulation import make_plot
from src.intensities import direct_intensity
import networkx as nx

import warnings
warn_list = ["divide by zero encountered in log",
             "divide by zero encountered in true_divide",
             "invalid value encountered in log",
             "invalid value encountered in true_divide"]
for warn in warn_list: warnings.filterwarnings("ignore", message=warn)
warnings.filterwarnings("ignore",category=DeprecationWarning)
import pickle
from os.path import exists
from os import makedirs
from datetime import datetime

# Return the true current parameter setting
def get_truth(S):
    return [f"{j}->{i}" for i,j in zip(*np.nonzero(S['adjacency'])) if i != j]

def get_params(pos, coord, order = None, double=False):
    """
    Get position of first and second order entries, relevant for procceeses in coord
    """
    first = (pos[0]!= -1) & (pos[1] == -1)
    second = (pos[1] != -1)
    relevant = np.isin(pos, coord)

    if order == 1: return np.unique(np.where(relevant & first)[1])
    elif order == 2:
        if double: return np.unique(np.where(np.all(relevant, axis=0))[0])
        else: return np.unique(np.where(relevant & second)[1])
    else: return np.unique(np.where(relevant)[1])


def evaluate_fit(S, fits, ts, pos, x_integrated, kappa, zoom=None, second_order=None, abC=None):
    """ Make figure showing fitted kernels

    Input
    - S: Settings dictionary
    - fits: Dictionary with optimization results
    - ts: Event history arrays
    - pos: Position arrays, indicating which parameters belong to which processes.
    - x_integrated: Vector of integrated x-intensity
    - kappa: Penalty parameter kappa
    - zoom: When plotting intensity, plot only the interval (0, zoom)
    - abC: Dictionary specifying a, b and C in a -> b | C
    - second_order: Indicator of whether second order fit is performed
    """
    basis1 = S["basis1"]
    kernel_support, run_time =  S["kernel_support"], S["run_time"]
    quadratic = S["quadratic"]
    if abC is None: abC = S['abC']
    if second_order is None: second_order = S['second_order']

    a, b, C = abC.values()
    # Evaluate fit
    # print("\nLikelihood function value")
    # for name, fit in fits.items(): print(f"{name}: {-fit['fun']}")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    grd = np.linspace(0, kernel_support, 101)
    fig, axs = plt.subplots(ncols=len(fits), sharey=True)
    for i, (name, fit) in enumerate(fits.items()):
        plt.sca(axs[i])
        axs[i].set_title(name)
        for j, (v, x) in enumerate({v: basis1.transform(grd).dot(fit['x'][get_params(pos, [v], 1)]) for v in [a] + C}.items()):
            # Plot fitted
            plt.plot(grd, x, label=f"{v}->{b}", alpha = (1 if v == a else 0.4), color=colors[j])

            # Plot true
            alpha, beta = S['adjacency'][b, v], S['decays'][b, v]
            plt.plot(grd, alpha*beta*np.exp(-beta*grd), ls="dotted", color=colors[j])



    plt.legend()
    fig.suptitle(f"Kernel plots (kappa = {kappa}, order = {1 + 1*second_order})")
    plt.show()

    # Plot along axis
    if zoom is None: grid = np.linspace(0, run_time, 1001)
    else: grid = np.linspace(0, zoom, 1001)

    x_g = direct_intensity(S, grid, ts, abC, pos=False, second_order=second_order)
    for name, fit in fits.items():
        plt.plot(grid, np.maximum(0, x_g.dot(fit['x'])), label=f"{name}")
    #TODO: Deprecated plot of true intensity, due to stop of tick use.
    # if tick_available:
    #     plt.plot(hawkes.intensity_tracked_times[::100], hawkes.tracked_intensity[b][::100], label='true')

    # Plot true intensity
    make_plot(S, ts, target=b, run_time=zoom)

    plt.legend()
    plt.title(f"Intensity plots (kappa = {kappa})")
    plt.show()

def pre_save_results(S):
    # Create folder
    folder_name = "results/"+ S["exp_name"]
    if not exists(folder_name + "/S_files"): makedirs(folder_name + "/S_files")

    # Check if cluster
    is_cluster = cpu_count() > 10
    c_string = '_cluster' if is_cluster else ''

    # Copy settings, update exp_id
    with open(f"results/_exp_id{c_string}.pkl", "rb") as file:
        exp_id = pickle.load(file)
    with open(f"results/_exp_id{c_string}.pkl", "wb") as file:
        pickle.dump(exp_id + 1, file)
    copyfile("settings.py", f"{folder_name}/S_files/exp{exp_id}{c_string}.py")

    return exp_id

def dummy_pos(S, second_order=None, abC=None):
    abC = S['abC'] if abC is None else abC
    second_order = S['second_order'] if second_order is None else second_order
    _, pos = direct_intensity(S, np.array([0]),
                              [np.array([]) for s in range(S["dims"])],
                              abC, second_order=second_order)
    return pos

class LciPrinter():
    """ Class to print LCI progress """
    def __init__(self, S, plot=False):
        self.S, self.plot = S, plot
    def p(self, b0=None,b1=None, b2=None):
        """ Print various stages """
        if not (self.S['plot'] or self.plot): return
        if b0 is not None: print('-'*20+'\n'+f"Kappa = {b0}"+'\n'+'-'*20)
        if b1 is not None: evaluate_fit(*b1)
        if b2 is not None: print(b2)


def save_results(S, abC, second_order, exp_id, results):
    a, b, C = abC.values()

    truth = '|'.join(get_truth(S))
    C_str = "C=("+",".join(map(str, C))+")"
    print(f"Truth: {truth}")
    print(f"Test: {a}->{b}|{C}")

    folder_name = "results/"+ S["exp_name"]

    is_cluster = cpu_count() > 10
    c_string = 'cluster' if is_cluster else ''

    save_str = f"exp{exp_id}_{c_string}_[{truth}]|SO={second_order}|{C_str}|" + datetime.now().strftime("%h%d-%H%M")
    with open(f"{folder_name}/{save_str}.pkl", 'wb') as output:
        pickle.dump(results, output)

# matrix implementation:
def sym(A):
    """Compute the subgraph of A containing only the 2-cycles of A"""
    return ((A + A.T) == 2).astype(int)


def shd(A, B, mistakes_as_one=True):
    """Compute the Structural Hamming Distance of two graphs, A and B

    Parameters
        ----------
        A, B : Either square matrices or networkx graph objects
        mistakes_as_one : bool
            Indicates whether reversals of edges should be counted as a single
            mistake. Defaults to true.
    """
    if "graph" in str(type(A)):
        A = nx.adjacency_matrix(A).todense()
    if "graph" in str(type(B)):
        B = nx.adjacency_matrix(B).todense()

    sd = (A + B) % 2

    if not mistakes_as_one:
        return np.sum(sd)

    B_ = (sym(sd) - ((sym(A) + sym(B)) > 0)) > 0
    A_ = (sd - B_) > 0

    return np.sum(A_) + np.sum(B_) / 2
