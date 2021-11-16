import numpy as np
from numpy.linalg import inv, pinv
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from src import omega, tools, likelihood, intensities, basis
from scipy.stats import chi2
from itertools import combinations as comb


class LIPP:
    def __init__(self, second_order=True, kappa=1e2,
                 phi_inv=lambda x: np.exp(np.minimum(x, 1) - 1)*(x<1) + x*(x>=1),
                 dphi_inv=lambda x: np.exp(np.minimum(x, 1)-1)*(x<1) + 1*(x>=1),
                 df1=8, df2=3, kernel_support=10,
                 Omega=None, omega_scale=10, size_grid=None, blocks=None,
                 abC=None):
        self.second_order = second_order
        self.kappa = kappa
        self.phi_inv = phi_inv
        self.dphi_inv = dphi_inv
        self.df1 = df1
        self.df2 = df2
        self.kernel_support = kernel_support
        self.Omega = None
        self.omega_scale = omega_scale
        self.blocks = blocks
        self.abC = abC

        self.basis1 = basis.Basis((-5, self.kernel_support), self.df1)
        self.basis2 = basis.Basis((-5, self.kernel_support), self.df2)
        self.size_grid = self.df1 - 2 if size_grid is None else size_grid

    def set_data(self, ts, run_time=None):
        self.ts = ts
        self.dims = len(ts)
        self.run_time = max(max(j) for j in ts) if run_time is None else run_time

    def set_hypothesis(self, abC):
        self.abC = abC

    def set_order(self, second_order):
        self.second_order = second_order

    def set_blocks(self):
        self.blocks = omega.get_omega_blocks(self.df1, self.df2, self.kernel_support, second_order=self.second_order)

    # Pre processing of experiment
    def optimization_vectors(self, abC=None):
        """ Compute x_dir, x_int  """
        abC = self.abC if abC is None else abC
        # Compute direct and integrated intensities
        a, b, C = abC.values()
        x_dir, pos = intensities.direct_intensity(self.basis1, self.basis2, self.kernel_support, self.ts[b], self.ts, abC, second_order=self.second_order)
        x_int, dx = intensities.integrated_outer_intensity(self.basis1, self.basis2, self.kernel_support, self.run_time, self.ts, abC, second_order=self.second_order)

        return x_dir, x_int, pos, dx


    def test_statistic(self, beta, kappa, Omega, pos, x_direct, ts, abC, plot=False, second_order=True):
        """ Function to get test statistic from estimate beta """
        run_time, kernel_support, basis1 = self.run_time, self.kernel_support, self.basis1
        size_grid = self.size_grid
        a = abC["a"]
        phi_inv, dphi_inv = self.phi_inv, self.dphi_inv

        # Get xx^T over a grid
        granularity = 4*int(run_time)
        grid        = np.linspace(0, run_time, granularity+1)
        x_g         = intensities.direct_intensity(self.basis1, self.basis2, self.kernel_support, grid, ts, abC, pos=False, second_order=second_order)
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

    def lci_test(self, kappa=None, Omega=None, lci_prep=None, abC=None):
        """ LCI test"""
        # Retrieve output
        kappa = self.kappa if kappa is None else kappa
        abC = self.abC if abC is None else abC

        # Compute vectors for optimization
        x_dir, x_int, pos, dx = self.optimization_vectors(abC=abC) if lci_prep is None else lci_prep
        self.Omega = omega.get_omega(self.df1, self.df2, self.kernel_support, pos, abC, scale=self.omega_scale, second_order=self.second_order) if Omega is None else Omega
        opt_arg = (x_dir, x_int, -1, dx, kappa, self.Omega, self.phi_inv, self.dphi_inv)


        # As initialization, fit constrained problem with positive parameters
        init = np.ones(x_int.shape[-1])/10000
        opt1 = minimize(likelihood.likelihood, init, args=opt_arg, method = 'L-BFGS-B', bounds = Bounds(0, np.inf), jac=likelihood.dlikelihood)
        # Then use solution to constrained problem to solve unconstrained
        opt2 = minimize(likelihood.likelihood, opt1['x'], args=opt_arg, method = 'L-BFGS-B', jac=likelihood.dlikelihood)
        T_a = self.test_statistic(opt2['x'], kappa, self.Omega, pos, x_dir, self.ts, abC, second_order=self.second_order)

        return {"Test statistic": T_a, "pval": 1-chi2(self.size_grid).cdf(T_a)}

    def eca(self, verbose=False):
        """ The actual ECA algorithm """
        if self.blocks is None:
            self.set_blocks()
        # Run ECA
        V = set(np.arange(self.dims))
        adj = np.zeros((self.dims, self.dims))
        for v in V:
            pa_v = V.copy()
            for v_prime in pa_v - {v}:
                d = 0
                while (d < len(pa_v)) & (v_prime in pa_v):
                    for C in comb(pa_v - {v, v_prime}, d): #(1)
                        if verbose: print(f"Testing {v_prime} -> {v}|{list(C) + [v]}")
                        abC = {"a": v_prime, "b": v, "C": list(C) + [v]} #(2)
                        pos = tools.dummy_pos(self.basis1, self.basis2, abC=abC, second_order=self.second_order, dims=self.dims)
                        Omega = omega.get_omega(self.df1, self.df2, self.kernel_support, pos, abC, self.blocks, second_order=self.second_order)
                        p_val = self.lci_test(Omega=Omega, abC=abC)['pval']
                        local_indep = (p_val >= 0.05)
                        if local_indep: # (2)
                            pa_v.discard(v_prime)
                            break
                    d += 1
            adj[v, list(pa_v)] = 1
        return adj
