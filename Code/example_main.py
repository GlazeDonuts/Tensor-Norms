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