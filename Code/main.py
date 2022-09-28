import argparse
import experiments
import utils

utils.set_seed(912)

num_samples = 5
# parser = argparse.ArgumentParser(description='Experiment')
# parser.add_argument('--dim_list', type=list, required=False, help='dim list', default=range(10,255,10))
# parser.add_argument('--order_list', type=list, required=False, help='order list', default=[2, 3])
# parser.add_argument('--alloc', type=float, required=True, help='total gpu memory allocated')
# parser.add_argument('--num_samples', type=int, required=True, help='total number of samples per dimension')
# parser.add_argument('--cmplx', type=bool, required=False, help='complex', default=False)
# args = parser.parse_args()

def prakrit():
	for normalize in [False, True]:
		for cmplx in [False, True]:
			experiments.GaussExp(dim_list=range(250, 0, -5), order_list=[3, 2], num_samples=num_samples, alloc=3.5, cmplx=cmplx, normalize=normalize)

def shitty():
	alloc_gpu = 3.5
	alloc_cpu = 13
	order_list = [3, 2]
	small_list = range(60, 0, -15)
	large_list = range(60, 0, -15)

	count = 0
	skip = 0

	for periodic in [True]:
		for rep in [True]:
			for cmplx in [True]:
				for normalize in [False]:
					if count < skip:
						print(f"Skipping Periodic: {periodic}, Repetition: {rep}, Complex: {cmplx}, Normalize: {normalize}")
					else:
						experiments.GaussMPSExp_DF(order_list, dim_list=small_list, bond_dim_list=large_list, num_samples=num_samples, periodic=periodic, rep=rep, cmplx=cmplx, normalize=normalize, alloc_gpu=alloc_gpu, alloc_cpu=alloc_cpu, batch_size=None)
					count += 1

def idk_who_yet():
    for periodic in [True, False]:
        for rep in [True, False]:
            for cmplx in [True, False]:
                experiments.NormedGaussMPSExp_BF(order_list, bond_dim_list=small_list, dim_list=large_list, num_samples=num_samples, periodic=periodic, rep=rep, cmplx=cmplx, alloc_gpu=3.5, alloc_cpu=13, batch_size=None)

# experiments.GaussExp(dim_list=args.dim_list, order_list=args.order_list, num_samples=args.num_samples, alloc=args.alloc, cmplx=args.cmplx)
# experiments.GaussExp(dim_list=range(10, 255, 10), order_list=[3, 2], num_samples=50, alloc=11.5, cmplx=False)
# experiments.GaussExp(dim_list=range(10, 135, 10), order_list=[4], num_samples=50, alloc=11.9, cmplx=False)

# experiments.GaussMPSExp(dim_list=range(255, 5, -10), order_list=[3,2], bond_dim_list=[3,2], num_samples=50, alloc=11.5, cmplx=True)
# experiments.GaussMPSExp(dim_list=range(255, 5, -10), order_list=[3,2], bond_dim_list=[3,2], num_samples=50, alloc=11.5, cmplx=False)

# shitty()

# experiments.GaussExp(dim_list=range(80, 60, -10), order_list=[4, 3, 2], num_samples=num_samples, alloc=3.2, cmplx=True, normalize=False)

experiments.DickeExp(10)
# experiments.AntisymExp(dim_list=range(1,9))