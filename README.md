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
where, ```<username>``` is the username on your system. The above output was obtained on a MacOS system, but the output should be similar on other systems as well.
  
### Execution instructions
- Change the current working directory to the ```Code``` folder using the following command.
```
cd Code
```
- Create a file ```example_main.py``` or ```main.py``` if not already present.
- Import modules as required.
- An example file is provided below and also included in the repository.

```
  import experiments
  import plotter
  '''
  Experiments:
  We first run a random real Gaussian state experiment for order 2 (bipartite) and 3 (tripartite) states with local dimensions 15, 25 and 35.

  Then, we run a random real Gaussian MPS experiment with bond dimensions 5, 10 and physical dimensions 15, 25 with 3 sites each, with translation invariance and periodic boundary condoitions.
  '''

  normalize = False

  gauss_cmplx = False
  mps_cmplx = False
  rep = True
  periodic = True

  gauss_order_list = [2, 3]
  gauss_dim_list = [15, 25, 35]

  mps_order_list = [3]
  mps_dim_list = [15, 25]
  mps_bond_dim_list = [5, 10]

  experiments.GaussExp(dim_list=gauss_dim_list, order_list=gauss_order_list, num_samples=4, device='cpu', batch_size=4, alloc=4, cmplx=gauss_cmplx, normalize=normalize)

  experiments.GaussMPSExp_DF(order_list=mps_order_list, dim_list=mps_dim_list, bond_dim_list=mps_bond_dim_list, num_samples=4, device='cpu', batch_size=4, alloc_cpu=4, periodic=periodic, rep=rep, cmplx=mps_cmplx, normalize=normalize)

  '''
  Plotting:
  We plot the results of the experiments above.
  The plot will be stored in a folder called "Plots/".
  If you also wish to see the plots, then set show=True in the plt_config class in Code/config.py.
  '''

  # # Plotting the results for the non-symmetrized Gaussian states (similar to Figure 3 in the paper)
  plotter.plot_non_gauss(f"Logs/Gaussian_{gauss_order_list}_{gauss_dim_list}_4_{gauss_cmplx}_{normalize}.json")

  # # Plotting the results for the symmetrized Gaussian states (similar to Figure 2 in the paper)
  plotter.plot_sym_gauss(f"Logs/Gaussian_{gauss_order_list}_[15, 25, 35]_4_{gauss_cmplx}_{normalize}.json")

  # Plotting the results for the MPS states (similar to Figure 8 in the paper), scaling with physical dimension
  plotter.plot_mps_bf(f"Logs/GaussianMPS_{mps_order_list}_periodic_{periodic}_rep_{rep}_cmplx_{mps_cmplx}_normalize_False_DF.json", periodic=periodic, rep=rep, cmplx=cmplx)
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