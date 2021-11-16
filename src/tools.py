import numpy as np
import networkx as nx
from src import intensities


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

def dummy_pos(basis1, basis2, second_order, abC, dims):
    _, pos = intensities.direct_intensity(basis1, basis2, kernel_support=10, time=np.array([0]),
                              ts=[np.array([]) for s in range(dims)],
                              abC=abC, second_order=second_order)
    return pos


# Matrix implementation:
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
