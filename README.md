# Tensor-Norms
Code for algorithms and experiments on the multipartite entanglement of random tensors. You can find our paper [here](https://arxiv.org/abs/2209.11754).
### Installation instructions
- Download the repository.
- Copy the code folder to the working directory.
- Delete ```main.py``` if already exists.
- Import modules as required.

**Note:** One can choose to use this repository as a library by simply copying the folder and accessing modules via ```import Tensor-Norms.<module_name> as <module_name>```.

### Module List

- ```wrapper.py``` Contains basic wrappers for the tensor and approximation classes.
- ```decomp.py``` Contains all tensor decomposition functions.
- ```experiments.py``` Contains experiments on random Gaussian tensors and random MPS.
- ```plotter.py``` Contains plotting functions corresponding to each experiment.
- ```utils.py``` Contains utility functions used in all of the above modules. Typically, this module would be important while designing new expermeints.

**Note:** Further, explainations are provided in the form of relevant comments in the code.
