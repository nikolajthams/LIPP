import numpy as np
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement as comb_rep
from src.tools import get_params

def scale_omega(Omega, omega_scale, pos, C):
    Omega_tmp = np.copy(Omega)
    idxC = get_params(pos, C, 2)
    Omega_tmp[idxC, idxC] *= omega_scale
    return Omega_tmp

def get_omega_blocks(S, granularity=1000, second_order=True):
    df1, df2, kernel_support = S["df1"], S["df2"], S["kernel_support"]
    # a, b, C = abC.values()
    def r_bs(df, derivs, intercept=False, plot=False):
        """Returns differentiated design matrices (using R-call via rpy2)"""
        base = importr('base')
        splines = importr('splines')

        # Setup ordinary B-spline to compute knots
        spl = splines.bs(base.c(0, kernel_support), df = df, intercept = intercept)
        knots = base.attributes(spl).rx2('knots')
        knots = base.c(base.rep(0, base.ifelse(intercept, 4, 3)), knots, base.rep(kernel_support, 4))

        # Compute design matrix
        grd = np.linspace(0, kernel_support, granularity + 1)
        x = FloatVector(grd)
        x = np.asarray(splines.splineDesign(x, knots = knots, outer_ok = True, derivs = derivs))

        if plot:
            plt.plot(grd, x)
            plt.title(f"Derivs {derivs}, df {df}")
            plt.show()

        return x

    def get_partial_block(deriv1, deriv2, diag):
        """Get the integrated partial blocks (i.e. order 1,1 or 2, 0)
        Integrand is e.g. vec(b'' b^T)
        """
        # Compute each spline matrix
        b1     = r_bs(df=df2, derivs=deriv1)
        b1     = np.einsum("ij, ik -> ijk", b1, np.ones(b1.shape))
        b2     = r_bs(df=df2, derivs=deriv2)
        b2     = np.einsum("ij, ik -> ijk", b2, np.ones(b2.shape))

        # Get the ordering of b1 and b2 right - depends on whether diag or non-diag
        if diag:
            upper_diag     = np.triu_indices(b1.shape[1])
            b1             = np.stack([x[upper_diag] for x in b1])
            b2             = np.stack([x.T[upper_diag] for x in b2])
        else:
            b1             = b1.reshape(b1.shape[0], -1)
            b2             = np.moveaxis(b2, 1, 2).reshape(b2.shape[0], -1)

        # Perform each integration separately
        outer1 = np.einsum("ij, ik -> ijk", b1, b1).sum(axis=0) * kernel_support/granularity
        outer2 = np.einsum("ij, ik -> ijk", b2, b2).sum(axis=0) * kernel_support/granularity

        return outer1 * outer2

    def get_block(diag):
        """Wrapper for getting f''g + 2f'g' + fg'' """
        out = (get_partial_block(2, 0, diag) +
               2*get_partial_block(1, 1, diag) +
               get_partial_block(0, 2, diag))
        return out

    block1 = r_bs(df=df1, derivs=2)
    block1 = np.einsum("ij, ik -> jk", block1, block1) * kernel_support/granularity

    # Diagonal and off-diagonal blocks
    block_d = get_block(True,)
    block_o = get_block(False)

    return block1, block_d, block_o

def get_omega(S, pos, abC, blocks=None, granularity=1000, second_order=True, scale = 10):
    a, b, C = abC.values()
    param_size = int(1 + len([a] + C)*S["df1"] + (((len(C)*S["df2"])**2 + len(C)*S["df2"])/2 if second_order else 0))

    if blocks is None:
        blocks = get_omega_blocks(S, granularity=granularity, second_order=second_order)

    block1, block_d, block_o = blocks
    # Initialize
    Omega = np.zeros((param_size, param_size))

    for v in [a] + C:
        idx = get_params(pos, v, order=1)
        Omega[np.ix_(idx, idx)] = block1

    if second_order:
        for c1, c2 in list(comb_rep(C, 2)):
            if c1 == c2: idx = get_params(pos, c1, order=2, double=True)
            else:
                idx1, idx2 = get_params(pos, c1, order=2), get_params(pos, c2, order=2)
                idx = np.array(list(set(idx1) & set(idx2)))

            Omega[np.ix_(idx, idx)] = scale*block_d if c1 == c2 else scale*block_o

    # if scale != 1:
    #     idxC = get_params(pos, C, 2)
    #     Omega[idxC, idxC] *= scale

    return Omega

def identity_omega(S, pos, abC, scale = 1):
    a, b, C = abC.values()
    Omega = np.identity(pos[0].size)
    pos_2 = get_params(pos, C, 2)
    Omega[np.ix_(pos_2, pos_2)] *= S['omega_scale']
    return Omega
