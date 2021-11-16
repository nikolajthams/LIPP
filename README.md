# LIPP: Local Independence in Point Processes
This code contains an implementation of the test for local independence presented in the paper [Local Independence Testing for Point Processes](https://arxiv.org/abs/2110.12709). 

The main class is the `LIPP()` class in `src/lci_test.py`. 

- In `example_lci_test.py`, we demonstrate how to concretely test a hypothesis and run the ECA structure learning algorithm. 
- In `example_turtle_data.py`, we run the structure learning algorithms on the turtle neuron firing data described in the paper. The results are analyzed in `analysis_turtle_data/analysis.py`. 
