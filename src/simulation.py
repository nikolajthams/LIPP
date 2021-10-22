import numpy as np
from numpy.random import exponential as e
from os.path import isfile
import pickle
import importlib
import matplotlib.pyplot as plt

# Tick library unavailable on some systems
tick_available = importlib.util.find_spec("tick") is not None
if tick_available:
    from tick.hawkes import SimuHawkesExpKernels
    from tick.plot import plot_point_process


def get_truth(S):
    """Return list of edges as strings"""
    return [f"{j}->{i}" for i,j in zip(*np.nonzero(S['adjacency'])) if i != j]


def get_hawkes(S, use_tick=False, plot=False, seed=None, n=None, adj=None):
    """Return sample from Hawkes process with specified parameters

    Input parameters
    - S: Settings dictionary
    - use_tick: Indication of whether to simulate using tick library or implementation of Ogata razor
    - plot: Boolean indicating whether to print plots
    - Seed: Simulation seed
    - n:
    """

    # Option to parse graph instead of loading from S_file
    adj = (adj if adj is not None else S['adjacency'])

    if not use_tick: return sim_hawkes(S, seed=seed, plot=plot, adj=adj)
    if tick_available: return use_tick(S, seed=seed, adj=adj, plot=plot)
    else:
        truth = "_".join(get_truth(S))
        location = f"sims_{truth}.pkl"
        if isfile(location):
            with open(f"sims_{truth}.pkl", 'rb') as input:
                container = pickle.load(input)
            return container[n]
        else:
            print("Library tick not found and no saved data for this truth available")

#### Sim from tick (linear Hawkes)
def use_tick(S, seed=None, adj=None, plot=False):
    run_time, dims = S["run_time"], S["dims"]
    hawkes = SimuHawkesExpKernels(adjacency=adj, decays=S["decays"],
                                  end_time = run_time, baseline=S["baseline"],
                                  verbose = False, seed=seed)

    # Make sure we don't get an explosive process
    if hawkes.spectral_radius() >= 0.9:
        print("Adjusting spectral radius")
        hawkes.adjust_spectral_radius(0.9)

    # Cache the intensity to plot
    hawkes.track_intensity(0.01)

    # Simulate
    hawkes.simulate()

    # Plotting
    if plot:
        fig, ax = plt.subplots(dims, 1, figsize=(16, 8), sharex=True, sharey=True)
        plot_point_process(hawkes, n_points=50000, t_min=10, max_jumps=30, ax=ax)
        fig.tight_layout()
        plt.show()

    return hawkes.timestamps

##### Sim from nonlinear Hawkes
def cut(ts, u, l=-np.Inf, weak=False):
    """Select events in the interval (l, u] or (l, u)"""
    return [x[(l<x) & (x <= u if weak else x < u)] for x in ts]

def load_par(S):
    """Help function to load parameters"""
    return S['adjacency'], S['decays'], S['baseline'], S['kernel_support']

def unif(u=1, l=0):
    """Random uniform"""
    return np.random.uniform(low=l, high=u)

def linear_intensity(t, ts, S, p=None, phi_inv=None, adj=None):
    """Return intensity of process based on past events with exponential kernel

    Input
    - t: Time at which to compute intensity
    - ts: Event history
    - S: Setting dictionary
    - p: Parameters to overwrite default parameters parameters in S
    - phi_inv: transformation of linear part
    """
    # Load parameters
    J = range(S['dims'])
    alpha, beta, mu, kernel_support = p if p is not None else load_par(S)
    alpha = (alpha if adj is None else adj)

    # Cut and compute likelihood. Likelihood is alpha*beta*exp(-beta*(t - t_i))
    ts = cut(ts, t, t - kernel_support)
    l = [np.dot((alpha*beta)[i], [np.exp(-beta[i,j]*(t - ts[j])).sum() for j in J]) for i in J]

    # return np.maximum(np.add(l, mu), 0)
    if phi_inv is None: phi_inv = S['phi_inv']
    return phi_inv(np.add(l, mu))

def upper_bound(t, ts, S, adj=None):
    """Upper bound function for accept/reject sampling in sim_hawkes"""
    p = load_par(S)

    p = (p[0] if adj is None else adj, ) + p[1:]

    # To use Ogata razor, we need non-increasing intensity.
    # If there are inhibitory effects, we need to upper bound without those
    p = (np.maximum(p[0], 0),) + p[1:]

    return linear_intensity(t, ts, S, p)

def make_plot(S, ts, target=None, run_time=None):
    """Plot events and intensity. Intensity is plotted for the target process, or all if target is None"""
    run_time = S['run_time'] if run_time is None else run_time
    x = np.linspace(0, run_time, int(100*run_time**0.5))
    if target is None:
        plt.plot(x, np.array(list(map(lambda t: linear_intensity(t, ts, S), x))))
        [plt.scatter(z, np.ones(z.shape)*-0.5, label=j) for j, z in enumerate(ts)]
    else:
        plt.plot(x, np.array(list(map(lambda t: linear_intensity(t, ts, S), x)))[:,target], label="True intensity", alpha=0.4)

def sim_hawkes(S, plot=False, seed=None, adj=None, track_intensity=False):
    """Simulate point process using Ogata razor.

    Input
    - S: Settings dictionary
    - plot: Indication of whether to plot events and intensities
    - seed: Seed for randomization.
    """

    # Extract parameters from S
    dims, max_points, run_time = S['dims'], S['max_points'], S['run_time']

    # Initiate
    t, ts = 0, [np.array([]) for d in range(dims)]

    # Return intensities
    tracked_intensity = [np.array([]) for d in range(dims)]

    # Ogata razor: Compute upper bound for intensity, simulate exponentially distributed waiting times and accept/reject sample points
    while sum(x.size for x in ts) < max_points:
        # Draw candidate event time
        M = upper_bound(t, ts, S, adj=adj)
        M = M.sum()
        t += e(1/M)
        if t > run_time: break

        # Accept/reject sample the point
        intn = linear_intensity(t, ts, S, adj=adj)
        if unif(M) < intn.sum():
            i = np.random.choice(S['dims'], p=intn/intn.sum())
            ts[i] = np.append(ts[i], t)
            if track_intensity:
                tracked_intensity[i] = np.append(tracked_intensity[i], intn[i])

    # Raise exception for too many points. This causes likelihood problems.
    if sum(x.size for x in ts) >= max_points:
        raise Exception("Max points reached before run time")

    if plot:
        make_plot(S, ts)
        plt.legend()
        plt.show()

    if track_intensity:
        return ts, tracked_intensity
    else: return ts
