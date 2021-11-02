import numpy as np
from numpy.linalg import inv, pinv
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from src import omega, tools, likelihood, intensities
from scipy.stats import chi2

# Pre processing of experiment
def optimization_vectors(S, ts, second_order, abC):
    """ Compute x_dir, x_int, constr """

    # Compute direct and integrated intensities
    a, b, C = abC.values()
    x_dir, pos = intensities.direct_intensity(S, ts[b], ts, abC, second_order=second_order)
    x_int, dx = intensities.integrated_outer_intensity(S, ts, abC, second_order=second_order)

    # Compute constraints
    constr_grid = np.concatenate((ts[b], np.linspace(0, S["run_time"], 2*int(S["run_time"]))))
    constr = intensities.direct_intensity(S, constr_grid, ts, abC, pos=False, second_order=second_order)

    # Pack optimization argument (only for src simplicity)
    opt_arg = (x_dir, x_int, -1, dx)

    return x_dir, x_int, constr, pos, dx, opt_arg

def test_statistic(S, beta, kappa, Omega, pos, x_direct, ts, abC, plot=False, second_order=True):
    """ Function to get test statistic from estimate beta """
    run_time, kernel_support, basis1 = S["run_time"], S["kernel_support"], S["basis1"]
    size_grid = S["size_grid"]
    a = abC["a"]
    phi_inv, dphi_inv = S['phi_inv'], S['dphi_inv']

    # Get xx^T over a grid
    granularity = 4*int(run_time)
    grid        = np.linspace(0, run_time, granularity+1)
    x_g         = intensities.direct_intensity(S, grid, ts, abC, pos=False, second_order=second_order)
    z           = np.einsum("ki, kj -> kij", x_g, x_g)
    z           = np.moveaxis(z,0, 2)


    #K is int xx^T *dphi^2/phi dt in mle and int xx^T * dphi^2*phi dt for quad
    lin_intensity = x_g.dot(beta)
    K = z*np.power(dphi_inv(lin_intensity), 2)*phi_inv(lin_intensity)

    K = K.sum(axis=2)*run_time/granularity

    #In J = K (minus penalty)
    J = K - 2*kappa*Omega
    J_inv   = inv(J)

    # Compute sandwich estimator J^{-1} K J^{-1}
    sigma = J_inv.dot(K).dot(J_inv)

    # Find the indices related to a
    pos_a       = tools.get_params(pos, [a])
    sigma_a     = sigma[pos_a][:,pos_a]
    beta_a      = beta[pos_a]

    # (Wood 2012)-approach: Evaluate bases in grid
    x_grid      = np.linspace(0, kernel_support, size_grid)
    B           = basis1.transform(x_grid)
    try:
        sigma_grid  = pinv(B.dot(sigma_a).dot(B.T))

        #TODO: Commented the max(0) out. This is on the parameter, not the intensity
        # And it is completely okay to have a negative kernel estimate.
        # grid_intensity = np.maximum(beta_a.dot(B.T), 0)
        grid_intensity = beta_a.dot(B.T)
        T_a         = grid_intensity.dot(sigma_grid).dot(grid_intensity)
    except np.linalg.LinAlgError:
        print('SVD not converged')
        T_a = None
    if plot:
        plt.title("Kernel tested to 0")
        plt.plot(x_grid, beta_a.dot(B.T))
        plt.show()

    return T_a

def lci_test(S, ts, second_order=None, kappa=None, Omega=None, lci_prep=None, abC=None):
    """ LCI test"""
    # Retrieve output
    second_order = S['second_order'] if second_order is None else second_order
    kappa = S['kappa'] if kappa is None else kappa
    abC = S['abC'] if abC is None else abC

    # Compute vectors for optimization
    lci_prep = optimization_vectors(S, ts, second_order, abC) if lci_prep is None else lci_prep
    x_dir, x_int, constr, pos, dx, _opt_arg = lci_prep
    Omega = omega.get_omega(S, tools.dummy_pos(S, second_order=second_order), abC, scale=S['omega_scale'], second_order=second_order)
    init, opt_arg = np.ones(x_int.shape[-1])/10000, (*_opt_arg, kappa, Omega, S['phi_inv'], S['dphi_inv'])

    # As initialization, fit constrained problem with positive parameters
    opt1 = minimize(likelihood.likelihood, init, args=opt_arg, method = 'L-BFGS-B', bounds = Bounds(0, np.inf), jac=likelihood.dlikelihood)
    # Then use solution to constrained problem to solve unconstrained
    opt2 = minimize(likelihood.likelihood, opt1['x'], args=opt_arg, method = 'L-BFGS-B', jac=likelihood.dlikelihood)
    T_a = test_statistic(S, opt2['x'], kappa, Omega, pos, x_dir, ts, abC, second_order=second_order)

    return {"Test statistic": T_a, "pval": 1-chi2(S["size_grid"]).cdf(T_a)}
