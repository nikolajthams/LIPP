import numpy as np
import pandas as pd
from src import lci_test

# Load data
df = pd.read_csv("example_data.csv")
dims = np.unique(df['id']).size
ts = [df[df['id'] == i]['t'].values for i in  pd.unique(df['id'])]

# Select hypothesis to test
abC = {'a': 0, 'b': 2, 'C': [1, 2]}

# Load class
Tester = lci_test.LIPP()
Tester.set_data(ts)

# Compute test of hypothesis 0 -> 2 | 1, 2
Tester.set_hypothesis(abC)
print(Tester.lci_test())

# Estimate causal graph
print(Tester.eca(verbose=True))
