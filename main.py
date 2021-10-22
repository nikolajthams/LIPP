import numpy as np
import pandas as pd
from src import basis, lci_test

###
# from src import simulation, lci_test, tools, omega, likelihood
# from scipy.optimize import minimize, Bounds
# import matplotlib.pyplot as plt
###


df = pd.read_csv("example_data.csv")
dims = np.unique(df['id']).size

ts = [df[df['id'] == i]['t'].values for i in  pd.unique(df['id'])]


# Hypothesis to be tested: a -/> b | C
abC = {'a': 0, 'b': 2, 'C': [1, 2]}



#SETTINGS: Specify settings for the test
S = {
    "dims":             len(ts),
    "run_time":         max(max(j) for j in ts),
    "abC":              abC,
    "kernel_support":   10,
    "df1":              8,
    "df2":              3,
    "second_order":     True,
    "kappa":            1e2,
    "omega_scale":      10,
    "phi_inv":          lambda x: np.exp(np.minimum(x, 1) - 1)*(x<1) + x*(x>=1),
    "dphi_inv":         lambda x: np.exp(np.minimum(x, 1)-1)*(x<1) + 1*(x>=1)
}
# Basis expansions used for first and second order test
S["basis1"] = basis.Basis((-5, S["kernel_support"]), S["df1"])
S["basis2"] = basis.Basis((-5, S["kernel_support"]), S["df2"])
S["size_grid"] = S['df1']-2

# Test local independence a -/> b | C
lci_test.lci_test(S, ts)
