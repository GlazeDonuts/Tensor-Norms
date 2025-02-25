# Tensor-Norms
Code for algorithms and experiments on the multipartite entanglement of random tensors. You can find our paper [here](https://arxiv.org/abs/2209.11754).

### Installation instructions
- Download the repository.
- Set the current working directory to the downloaded repository.
- Create an environment as per the instructions below.

### Environment creation and activation
- Install Anaconda from this [link](https://www.anaconda.com/download).
- Open a new terminal window in the working directory (the main folder of the repository).
- Create a new environment using the ```environment.yml``` file using the code given below.
```
conda env create -n tensor_norms -f environment.yml
```
- The environment will most likely be activated automatically--at some point in the installation process, Ananonda asks you if it should auto-activate the environment. If not, please activate the environment using the code given below.
```
conda activate tensor_norms
```

### Sanity checks
1. Run the following code to make sure that the working directory is correct and that the environment is active.
```
ls
```
The output should list the following files and folders.
```
|- Code/
|- LICENSE
|- README.md
```
2. Further, run the following code to check if the environment is active.
```
conda env list
```
The output should list the ```tensor_norms``` environment with a ```*``` next to it, indicating that it is the active environment like seen in the example below.
```
tensor_norms          *  /Users/<username>/anaconda3/envs/tensor_norms
```
where, ```<username>``` is the username on your system. The above output was obtined on a MacOS system, but the output should be similar on other systems as well.
  
### Execution instructions
- Create a ```main.py``` if not already present.
- Import modules as required.
- An example file is provided below and also in the code.

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
- ```utils.py``` Contains utility functions used in all of the above modules. Typically, this module would be important while designing new experiments.
- ```environment.yml``` Is the environment file to create the environment.
We also provide an ```example_main.py``` file as an example file demonstrating how to run the experiments.


**_Note:_** Further explanations are provided in the form of relevant comments in the code.


_Pleasae feel free to write to us or report issues here on GitHub :)_