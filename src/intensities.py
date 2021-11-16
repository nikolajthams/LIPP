import numpy as np

def truncate(ts, upper, lower=-np.Inf, weak=False, multiple=None):
    """Upper truncate each coordinate of ts at time upper and lower."""
    # If nothing provided, check if ts is list (of np.) or np array
    if multiple is None: multiple = not isinstance(ts, np.ndarray)
    # If a list provided, run single-array function for each object
    if multiple: return [truncate(x, upper, lower, weak, multiple=False) for x in ts]
    return ts[(lower <= ts) & (ts <= upper if weak else ts < upper)]



def first_intensity(s, ts, keys, basis, kernel_support, pos=False):
    """ Computes intensity for some time s, and past points ts"""
    # The set of jumps affecting intensity at time s
    ts_rel = [ts[v] for v in keys]
    past_jumps = truncate(ts_rel, s, s - kernel_support)
    if isinstance(ts, np.ndarray): return basis.transform(s - past_jumps).sum(axis = 0)
    else:
        if not pos:
            return np.concatenate([basis.transform(s - x).sum(axis = 0) for x in past_jumps])
        else:
            return np.concatenate([basis.transform(s - x).sum(axis = 0) for x in past_jumps]), np.repeat(keys, basis.ncols)

def first_intensities(ss, ts, keys, basis, kernel_support, pos=False):
    """ Wrapper for computing intensities at several times"""
    if not pos:
        return np.stack([first_intensity(s, ts, keys, basis, kernel_support) for s in ss])
    else:
        return np.stack([first_intensity(s, ts, keys, basis, kernel_support) for s in ss]), np.repeat(keys, basis.ncols)

def direct_intensity(basis1, basis2, kernel_support, time, ts, abC, second_order=True, pos=True):
    """ At each point in ts[b] computes the intensity vector"""

    # Initiate series
    a, b, C = abC.values()
    aC      = np.unique([a] + C)

    # Test dummy case
    if time.size == 0: return None

    # 0: Constant intercept
    zeroth      = np.ones((len(time), 1))
    pos0        = np.array([[-1],[-1]])

    # 1: First order term
    first, pos1 = first_intensities(time, ts, aC, basis1, kernel_support, pos=True)
    pos1        = np.stack((pos1, -np.ones(pos1.size)))

    # 2: Second order term
    if (not second_order) or len(C) == 0:
        if pos: return np.concatenate((zeroth, first), axis=1), np.concatenate((pos0, pos1), axis=1)
        else: return np.concatenate((zeroth, first), axis=1)
    else:
        # This corresponds to first order effect of C only, but in basis2
        first2, pos2    = first_intensities(time, ts, C, basis2, kernel_support,pos=True)
        # Data
        outer           = np.einsum("ki, kj -> kij", first2, first2)
        upper_diag      = np.triu_indices(first2.shape[1])
        second          = np.stack([x[upper_diag] for x in outer])
        # Positions
        pos2            = np.tile(pos2, (len(pos2),1))
        pos2            = np.stack((pos2[upper_diag], (pos2.T)[upper_diag]))

    if pos: return np.concatenate((zeroth, first, second), axis=1), np.concatenate((pos0, pos1, pos2), axis=1)
    else:   return np.concatenate((zeroth, first, second), axis=1)

def integrated_intensity(basis1, basis2, kernel_support, time, ts, abC, second_order=True):

    # Initiate series
    ts      = truncate(ts, time)
    a, b, C = abC.values()
    aC      = np.unique([a] + C)

    ### Order 0
    zeroth = np.array([time])

    ### Order 1
    # Setup
    granularity = 1000
    step_size = (kernel_support - 0)/granularity
    grid = np.linspace(0, kernel_support, granularity+1)

    # Basic trapezoidal rule
    integrator = basis1.transform(grid)*step_size
    integrator = integrator.cumsum(axis=0) - integrator/2 - integrator[0]/2

    # For each event at distance time - ts, find nearest index in grid.
    idx = [np.minimum(np.searchsorted(grid, time - ts[v], 'right'), len(grid)-1) for v in aC]
    first = np.concatenate([integrator[id].sum(axis=0) for id in idx])

    if (not second_order) or len(C) == 0: return np.concatenate((zeroth,first))

    ### Order 2
    # Setup
    granularity = 3000
    grid        = np.linspace(0, time, granularity+1)

    """
    We are taking the dt integral over the pair-effects. So we pick a grid,
    and compute the pair-intensity at each grid-point. Then we dot-out time axis and normalize
    """
    def first_wrap(s): return first_intensity(s, ts, C, basis2, kernel_support)
    # pool            = Pool()
    second          = np.stack(list(map(first_wrap, grid)))
    # pool.close()

    second[[1,-1]] /= 2 # Trapezoidal correction (tiny probably)
    second          = second.T.dot(second)*(time-0)/granularity # This is the integration step
    upper_diag      = np.triu_indices(second.shape[1])
    second          = second[upper_diag]
    return np.concatenate((zeroth, first, second))

def integrated_outer_intensity(basis1, basis2, kernel_support, run_time, ts, abC, second_order=True, gran_factor=2):
    ts = truncate(ts, run_time)
    a, b, C = abC.values()
    granularity = int(run_time)*gran_factor
    grid = np.linspace(0, run_time, granularity)
    x = direct_intensity(basis1, basis2, kernel_support, grid, ts, abC, second_order=second_order, pos=False)
    x[[0, -1]] /= 2
    dx = (run_time/granularity)
    # x = np.einsum("ij,ik -> jk", x, x)*(run_time/granularity)
    return x, dx
