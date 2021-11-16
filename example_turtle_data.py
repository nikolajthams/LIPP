import pandas as pd
from src import lci_test
from tqdm import tqdm
import pickle

# Load data
start, finish = 9.468, 19.468
df = pd.read_csv("spikes2.txt", sep=" ")
df["markTypeRaw"] = df.markType.str.split("_", expand=True)[0]
df["stimulus"] = (df["time"] >= start) & (df["time"] <= finish)
dims = 6

# Placeholders for results
results1, results2 = [], []

# Initialize testers
Tester_order1 = lci_test.LIPP(second_order=False)
Tester_order2 = lci_test.LIPP(second_order=True)

# Compute penalization once, instead of in loop
Tester_order1.set_blocks()
Tester_order2.set_blocks()

# Loop over experiments
for repetition in tqdm(range(1, 6)):
    # Subset data
    ts = df[(df['id'] == repetition) & df["stimulus"]]
    ts = [ts[ts['channel'] == j]['time'].values for j in pd.unique(ts['channel'])]
    # Select only dims
    ts = ts[:dims]
    # Set to start at t=0
    ts = [x - start for x in ts]

    Tester_order1.set_data(ts)
    Tester_order2.set_data(ts)

    # Runc first and second order eca
    adj1 = Tester_order1.eca(verbose=True)
    adj2 = Tester_order2.eca(verbose=True)

    results1.append(adj1)
    results2.append(adj2)

# Save results for analysis.
with open("analysis_turtle_data/res1.pkl", 'wb') as file:
    pickle.dump(results1, file)
with open("analysis_turtle_data/res2.pkl", 'wb') as file:
    pickle.dump(results2, file)
