# Tensor-Norms
Code for algorithms and experiments on the multipartite entanglement of random tensors. You can find our paper [here](https://arxiv.org/abs/2209.11754).
### Installation instructions
- Download the repository.
- Copy the code folder to the working directory.

### Execution instructions
- Create a ```main.py``` if not already present.
- Import modules as required.
- An example file is provided below.

  ```
  import experiments

  experiments.GaussExp(dim_list=[15, 25, 35], order_list=[2, 3], num_samples=500)

  '''
  Explanation:
  We run a random Gaussian state experiment for order 2 (bipartite) and 3 (tripartite) states with local dimensions 15, 25 and 35.
  The num_samples parameter controls how many random samples we average over for each (dimension, order) pair.
  
  '''
  ```

### Module List

- ```wrapper.py``` Contains basic wrappers for the tensor and approximation classes.
- ```decomp.py``` Contains all tensor decomposition functions.
- ```experiments.py``` Contains experiments on random Gaussian tensors and random MPS.
- ```plotter.py``` Contains plotting functions corresponding to each experiment.
- ```config.py``` Contains configuration variables for experiments and plotter functions.
- ```utils.py``` Contains utility functions used in all of the above modules. Typically, this module would be important while designing new expermeints.

**Note:** Further explainations are provided in the form of relevant comments in the code.
