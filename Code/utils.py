import csv
from itertools import permutations
import json
from sympy.combinatorics.permutations import Permutation
import math
import numpy as np
import random
from scipy.optimize import curve_fit
import torch
from torch import optim
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from tqdm import tqdm
import sys

import torch

from comp import CandyComp
from wrapper import CompSetList, Tensaur

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''
============================================
Utility Functions
============================================
'''
#region

def set_seed(seed=4444):
    SEED = seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return (elapsed_mins, elapsed_secs)


def normalize_compsetlist(comp_set_list, dim=-1):
    if not isinstance(comp_set_list, CompSetList):
        comp_set_list = CompSetList(comp_set_list)
    return CompSetList([x/torch.norm(x, dim=-1).unsqueeze(-1).repeat((1,)*(len(x.shape)-1)+(x.shape[-1],)) for x in comp_set_list.data_])


def my_criterion(a, b):
    return torch.norm(a - b)**2


def make_batch_list(temp_space, num_samples, alloc, device='cuda', cmplx=False):
    if cmplx == False:
        fac = 5
    else:
        fac = 3
    alloc = alloc * (1e9)
    batch_size = int(alloc // (fac*temp_space))
    num_batches = num_samples//batch_size    
    last_batch_size = num_samples - num_batches*batch_size
    if last_batch_size == 0:
        return [int(batch_size)]*num_batches
    return [int(batch_size)]*num_batches + [int(last_batch_size)]


def make_batch_list_mps(order, bond_dim, dim, temp_space, num_samples, alloc_gpu, alloc_cpu, device='cuda', cmplx=False):
    if cmplx == False:
        fac = 4
    else:
        fac = 3
    alloc_gpu = alloc_gpu * (1e9)
    alloc_cpu = alloc_cpu * (1e9)
    batch_size_1 = int(alloc_gpu // (fac*temp_space))

    a = torch.randn([dim]*(order+2))
    temp_space_cpu = sys.getsizeof(a.storage()) * 2
    batch_size_2 = int(alloc_cpu // temp_space_cpu)
    del a

    a = torch.randn([bond_dim]*(order+2))
    temp_space_cpu = sys.getsizeof(a.storage()) * 2
    batch_size_3 = int(alloc_cpu // temp_space_cpu)
    del a

    batch_size = np.min([batch_size_1, batch_size_2, batch_size_3])

    num_batches = int(num_samples//batch_size)    
    last_batch_size = num_samples - num_batches*batch_size
    if last_batch_size == 0:
        return [int(batch_size)]*num_batches
    return [int(batch_size)]*num_batches + [int(last_batch_size)]


def cyclic_perms(ls):
    ls = list(ls)
    perms = []
    for idx in range(len(ls)):
        perms.append(ls[-idx:]+ls[:-idx])

    return perms


def symmetrize(T, batched=False):

    if not isinstance(T, Tensaur):
        T = Tensaur(T, batched=batched)
    
    dim = T.data_.shape[-1]
    if batched:
        order = len(T.data_.shape) - 1
    else:
        order = len(T.data_.shape)
    
    shape_tuple = (dim,)*order
    if batched:
        shape_tuple = (T.batch_size,) + shape_tuple

    if shape_tuple != T.data_.shape:
        raise ValueError(f"Expected a tensor with shape [d, d, ... , d] but got {T.data_.shape}")

    norm_factor = math.factorial(order)

    indices = list(range(order))
    perm_inds = permutations(indices)
    
    count = 0
    out = None
    for perm in perm_inds:
        if batched:
            perm = tuple([0]+[x+1 for x in perm])
        count += 1
        if out is None:
            out = T.data_.clone()
        else:
            out += T.data_.clone().permute(perm)
            
    assert norm_factor == count
    out /= norm_factor
    return Tensaur(out, batched=batched)


def par_symmetrize(T, batched=False):

    if not isinstance(T, Tensaur):
        T = Tensaur(T, batched=batched)
    
    dim = T.data_.shape[-1]
    if batched:
        order = len(T.data_.shape) - 1
    else:
        order = len(T.data_.shape)
    
    shape_tuple = (dim,)*order
    if batched:
        shape_tuple = (T.batch_size,) + shape_tuple

    if shape_tuple != T.data_.shape:
        raise ValueError(f"Expected a tensor with shape [d, d, ... , d] or [batch, d, d, ... , d] but got {T.data_.shape}")

    norm_factor = order

    indices = list(range(order))
    perm_inds = cyclic_perms(indices)
    
    count = 0
    out = None
    for perm in perm_inds:
        if batched:
            perm = tuple([0]+[x+1 for x in perm])
        count += 1
        if out is None:
            out = T.data_.clone()
        else:
            out += T.data_.clone().permute(perm)
            
    assert norm_factor == count
    out /= norm_factor
    return Tensaur(out, batched=batched)


def swap_bf_df(logger):
    out = {}
    out['exp_start_time'] = logger['exp_start_time'] 
    out['alloc_gpu'] = logger['alloc_gpu']
    out['alloc_cpu'] = logger['alloc_cpu']
    out['complex'] = logger['complex']

    out['order_list'] = logger['order_list']
    for order in out['order_list']:
        out[str(order)] = {}
        out[str(order)]['bond_dim_list'] = logger[str(order)][str(logger[str(order)]['dim_list'][0])]['bond_dim_list']
        for bond_dim in out[str(order)]['bond_dim_list']:
            out[str(order)][str(bond_dim)] = {}
            out[str(order)][str(bond_dim)]['dim_list'] = logger[str(order)]['dim_list']
    
    for order in out['order_list']:
        for bond_dim in out[str(order)]['bond_dim_list']:
            if logger['complex'] == False:
                out[str(order)][str(bond_dim)]['avg_non_als_gm_list'] = []
                out[str(order)][str(bond_dim)]['avg_par_als_gm_list'] = []
                out[str(order)][str(bond_dim)]['avg_sym_als_gm_list'] = []
            out[str(order)][str(bond_dim)]['avg_non_ngd_gm_list'] = []
            out[str(order)][str(bond_dim)]['avg_par_ngd_gm_list'] = []
            out[str(order)][str(bond_dim)]['avg_sym_ngd_gm_list'] = []
            out[str(order)][str(bond_dim)]['avg_sym_sgd_gm_list'] = []
            out[str(order)][str(bond_dim)]['avg_sym_pow_gm_list'] = []
            for dim in out[str(order)][str(bond_dim)]['dim_list']:
                out[str(order)][str(bond_dim)][str(dim)] = {}
                out[str(order)][str(bond_dim)][str(dim)]['non_l2_norm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['non_l2_norm_list']
                out[str(order)][str(bond_dim)][str(dim)]['par_l2_norm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['par_l2_norm_list']
                out[str(order)][str(bond_dim)][str(dim)]['sym_l2_norm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['sym_l2_norm_list']

                if out['complex'] == False:
                    out[str(order)][str(bond_dim)][str(dim)]['non_als_gm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['non_als_gm_list']
                    out[str(order)][str(bond_dim)][str(dim)]['par_als_gm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['par_als_gm_list']
                    out[str(order)][str(bond_dim)][str(dim)]['sym_als_gm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['sym_als_gm_list']

                out[str(order)][str(bond_dim)][str(dim)]['non_ngd_gm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_gm_list']
                out[str(order)][str(bond_dim)][str(dim)]['par_ngd_gm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['par_ngd_gm_list']
                out[str(order)][str(bond_dim)][str(dim)]['sym_ngd_gm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['sym_ngd_gm_list']
                out[str(order)][str(bond_dim)][str(dim)]['sym_sgd_gm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['sym_sgd_gm_list']
                out[str(order)][str(bond_dim)][str(dim)]['sym_pow_gm_list'] = logger[str(order)][str(dim)][str(bond_dim)]['sym_pow_gm_list']

                out[str(order)][str(bond_dim)][str(dim)]['non_ngd_loss_list'] = logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_loss_list']
                out[str(order)][str(bond_dim)][str(dim)]['par_ngd_loss_list'] = logger[str(order)][str(dim)][str(bond_dim)]['par_ngd_loss_list']
                out[str(order)][str(bond_dim)][str(dim)]['sym_ngd_loss_list'] = logger[str(order)][str(dim)][str(bond_dim)]['sym_ngd_loss_list']
                out[str(order)][str(bond_dim)][str(dim)]['sym_sgd_loss_list'] = logger[str(order)][str(dim)][str(bond_dim)]['sym_sgd_loss_list']
                out[str(order)][str(bond_dim)][str(dim)]['sym_pow_loss_list'] = logger[str(order)][str(dim)][str(bond_dim)]['sym_pow_loss_list']

            if out['complex'] == False:
                out[str(order)][str(bond_dim)]['avg_non_als_gm_list'].append(float(np.mean(out[str(order)][str(bond_dim)][str(dim)]['non_als_gm_list'])))
                out[str(order)][str(bond_dim)]['avg_par_als_gm_list'].append(float(np.mean(out[str(order)][str(bond_dim)][str(dim)]['par_als_gm_list'])))
                out[str(order)][str(bond_dim)]['avg_sym_als_gm_list'].append(float(np.mean(out[str(order)][str(bond_dim)][str(dim)]['sym_als_gm_list'])))
            
            out[str(order)][str(bond_dim)]['avg_non_ngd_gm_list'].append(float(np.mean(out[str(order)][str(bond_dim)][str(dim)]['non_ngd_gm_list'])))
            out[str(order)][str(bond_dim)]['avg_par_ngd_gm_list'].append(float(np.mean(out[str(order)][str(bond_dim)][str(dim)]['par_ngd_gm_list'])))
            out[str(order)][str(bond_dim)]['avg_sym_ngd_gm_list'].append(float(np.mean(out[str(order)][str(bond_dim)][str(dim)]['sym_ngd_gm_list'])))
            out[str(order)][str(bond_dim)]['avg_sym_sgd_gm_list'].append(float(np.mean(out[str(order)][str(bond_dim)][str(dim)]['sym_sgd_gm_list'])))
            out[str(order)][str(bond_dim)]['avg_sym_pow_gm_list'].append(float(np.mean(out[str(order)][str(bond_dim)][str(dim)]['sym_pow_gm_list'])))
            
    # out['exp_end_time'] = logger['exp_end_time']

    return out


def mps_candidate(x, c_1, c_2, c_3, c_4, c_5, c_6):
    x1, x2 = x
    return (c_1/np.sqrt(x1) + c_2/np.sqrt(x2) + c_3/(np.sqrt(x1)*np.sqrt(x2)) + c_4/x1 + c_5/x2 + c_6/(x1*x2))**2


def gauss_candidate(x, c1, c2):
    return c1 + c2/np.sqrt(x)


def mps_fitter(path, order_list=None):
    f = open(path,)
    logger = json.load(f)
    f.close()

    if order_list is None:
        order_list = logger['order_list']

    for order in order_list:
        X1 = np.zeros(len(logger[str(order)]['dim_list'])**2)
        X2 = np.zeros(len(logger[str(order)]['dim_list'])**2)
        Y = np.zeros(len(logger[str(order)]['dim_list'])**2)
        count = 0
        for d_idx, dim in enumerate(logger[str(order)]['dim_list']):
            for b_idx, bond_dim in enumerate(logger[str(order)][str(dim)]['bond_dim_list']):
                X1[count] = dim
                X2[count] = bond_dim
                Y[count] = np.mean(np.divide(logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_gm_list'], logger[str(order)][str(dim)][str(bond_dim)]['non_l2_norm_list']))
                count += 1

        params, _ = curve_fit(mps_candidate, (X1, X2), Y, maxfev=int(1e6))
        return params


def mps_fitter_bf(logger, order_list=None):

    if order_list is None:
        order_list = logger['order_list']

    for order in order_list:
        X1 = np.zeros(len(logger[str(order)]['bond_dim_list'])**2)
        X2 = np.zeros(len(logger[str(order)]['bond_dim_list'])**2)
        Y = np.zeros(len(logger[str(order)]['bond_dim_list'])**2)
        count = 0
        for b_idx, bond_dim in enumerate(logger[str(order)]['bond_dim_list']):
            for d_idx, bond_dim in enumerate(logger[str(order)][str(bond_dim)]['dim_list']):
                X1[count] = bond_dim
                X2[count] = dim
                Y[count] = np.mean(np.divide(logger[str(order)][str(bond_dim)][str(bond_dim)]['non_ngd_gm_list'], logger[str(order)][str(bond_dim)][str(bond_dim)]['non_l2_norm_list']))
                count += 1

        params, _ = curve_fit(mps_candidate, (X1, X2), Y, maxfev=int(1e6))
        return params


def gauss_fitter(path, order=3, sym=True):
    f = open(path,)
    logger = json.load(f)
    f.close()

    if sym:
        prefix = "sym"
    else:
        prefix = "non"

    X = np.zeros(len(logger[str(order)]['dim_list']))
    Y = np.zeros(len(logger[str(order)]['dim_list']))
    for d_idx, dim in enumerate(logger[str(order)]['dim_list']):
        X[d_idx] = dim
        Y[d_idx] = np.mean(logger[str(order)][str(dim)][prefix+'_ngd_gm_list'])

    params, _ = curve_fit(gauss_candidate, X, Y, maxfev=int(1e6))
    return params


#endregion


'''
============================================
Old Utility Functions
============================================
'''

#region
def sym_gauss_ratio_maker(path, num_samples=10, factor=1, order_list=None):
    f = open(path,)
    data = json.load(f)
    f.close()
    count = 0
    sgd = data['sgd']
    ana_gm_list = [2, 2.343335, 2.53722, 2.67002, 2.76997, 2.84957, 2.91541, 2.97135, 3.01986]
    if order_list is None:
        order_list = data['order_list']
    for order in order_list:
        norm_list = []
        dim_list = data[str(order)]['dim_list']
        ana_gm = ana_gm_list[order-2]
        for dim in dim_list:
            norm_sum = 0
            for _ in range(num_samples):
                norm_sum += torch.norm(make_SymGauss(dim, order))
            norm_list.append(norm_sum/num_samples)
        ana_list = np.divide(np.ones(len(dim_list))*ana_gm, np.array(norm_list))
        als_list = np.divide(np.array(data[str(order)]['avg_als_gm_list']), np.array(norm_list))
        pow_list = np.divide(np.array(data[str(order)]['avg_pow_gm_list']), np.array(norm_list))
        if sgd:
            sgd_list = np.divide(np.array(data[str(order)]['avg_sgd_gm_list']), np.array(norm_list))
            ssd_list = np.divide(np.array(data[str(order)]['avg_ssd_gm_list']), np.array(norm_list))
        
        data[str(order)]['avg_ana_ratio_list'] = list(factor*ana_list)
        data[str(order)]['avg_als_ratio_list'] = list(factor*als_list)
        data[str(order)]['avg_pow_ratio_list'] = list(factor*pow_list)
        if sgd:
            data[str(order)]['avg_sgd_ratio_list'] = list(factor*sgd_list)
            data[str(order)]['avg_ssd_ratio_list'] = list(factor*ssd_list)
    
    dump_object = json.dumps(data)
    with open("Code/jsons/SymGaussian_Ratios_" + str(list(order_list)) + "_.json", "w") as outfile:
        outfile.write(dump_object)


def gauss_ratio_maker(path, num_samples=10, factor=1, order_list=None):
    f = open(path,)
    data = json.load(f)
    f.close()
    count = 0
    sgd = data['sgd']
    ana_gm_list = [2, 2.343335]
    if order_list is None:
        order_list = data['order_list']
    for order in order_list:
        norm_list = []
        dim_list = data[str(order)]['dim_list']
        ana_gm = ana_gm_list[order-2]
        for dim in dim_list:
            norm_sum = 0
            for _ in range(num_samples):
                norm_sum += torch.norm(make_Gauss(dim, order))
            norm_list.append(norm_sum/num_samples)
        if order==2:
            ana_list = np.divide(np.ones(len(dim_list))*ana_gm, np.array(norm_list))
        als_list = np.divide(np.array(data[str(order)]['avg_als_gm_list']), np.array(norm_list))
        if sgd:
            sgd_list = np.divide(np.array(data[str(order)]['avg_sgd_gm_list']), np.array(norm_list))
        
        if order==2:
            data[str(order)]['avg_ana_ratio_list'] = list(factor*ana_list)
        data[str(order)]['avg_als_ratio_list'] = list(factor*als_list)
        if sgd:
            data[str(order)]['avg_sgd_ratio_list'] = list(factor*sgd_list)
    
    dump_object = json.dumps(data)
    with open("Code/jsons/Gaussian_Ratios_" + str(list(order_list)) + "_.json", "w") as outfile:
        outfile.write(dump_object)


def mixgauss_ratio_maker(path, num_samples=10, factor=1, order_list=None):
    f = open(path,)
    data1 = json.load(f)
    f.close()

    f = open(path.replace('Gauss', 'SymGauss'),)
    data2 = json.load(f)
    f.close()

    count = 0
    sgd = data1['sgd']
    data = data1

    if order_list is None:
        order_list = data1['order_list']
    for order in order_list:
        norm_list1 = []
        norm_list2 = []
        dim_list = data1[str(order)]['dim_list']
        idx_list = []
        for idx in range(len(data2[str(order)]['dim_list'])):
            if data2[str(order)]['dim_list'][idx] in dim_list:
                idx_list.append(idx)

        for dim in dim_list:
            norm_sum1 = 0
            norm_sum2 = 0
            for _ in range(num_samples):
                norm_sum1 += torch.norm(make_Gauss(dim, order))
                norm_sum2 += torch.norm(make_SymGauss(dim, order))
            norm_list1.append(norm_sum1/num_samples)
            norm_list2.append(norm_sum2/num_samples)

        als_list1 = np.divide(np.array(data1[str(order)]['avg_als_gm_list']), np.array(norm_list1))
        als_list2 = np.divide(np.array(data2[str(order)]['avg_als_gm_list'])[idx_list], np.array(norm_list2))
        if sgd:
            sgd_list1 = np.divide(np.array(data1[str(order)]['avg_sgd_gm_list']), np.array(norm_list1))
            sgd_list2 = np.divide(np.array(data2[str(order)]['avg_sgd_gm_list'])[idx_list], np.array(norm_list2))
        
        
        data[str(order)]['avg_als_ratio_list'] = list(factor*np.divide(als_list1, als_list2))
        if sgd:
            data[str(order)]['avg_sgd_ratio_list'] = list(factor*np.divide(sgd_list1, sgd_list2))
    
    dump_object = json.dumps(data)
    with open("Code/jsons/MixGaussian_Ratios_" + str(list(order_list)) + "_.json", "w") as outfile:
        outfile.write(dump_object)

  
#endregion

"""
============================================
Old Decomps
============================================
"""
#region

# not needed
class Cores(nn.Module):

    def __init__(self, shape_tuple, rank=1, cmplx=False):
        super(Cores, self).__init__()
        self.rank = rank

        self.normalized_list = nn.ModuleList()
        for shape in shape_tuple:
            m = torch.nn.utils.weight_norm(nn.Embedding(rank, shape), name='weight')
            m.weight_g = torch.nn.Parameter(torch.ones(m.weight_g.shape))
            self.normalized_list.append(m)
    
    def forward(self, device):
        approx = None
        
        for r in range(self.rank):
            comp_temp = self.normalized_list[0](torch.LongTensor([r]).to(device)).view(-1)
            for module in self.normalized_list[1:]:
                comp_temp = comp_temp.unsqueeze(-1) @ (module(torch.LongTensor([r]).to(device))).view(1, -1)
            if approx is None:
                approx = comp_temp
            else:
                approx += comp_temp
        
        return approx

# not needed
class SymCores(nn.Module):

    def __init__(self, shape_tuple, rank=1, cmplx=False):
        super(SymCores, self).__init__()
        self.rank = rank

        self.normalized_list = nn.ModuleList()
        shape = shape_tuple[0]
        self.order = len(shape_tuple)
        self.m = torch.nn.utils.weight_norm(nn.Embedding(rank, shape), name='weight')
        self.m.weight_g = torch.nn.Parameter(torch.ones(self.m.weight_g.shape))
    
    def forward(self, device):
        approx = None

        for r in range(self.rank):
            comp_temp = self.m(torch.LongTensor([r]).to(device)).view(-1)
            for _ in range(self.order-1):
                comp_temp = comp_temp.unsqueeze(-1) @ (self.m(torch.LongTensor([r]).to(device))).view(1, -1)
            if approx is None:
                approx = comp_temp
            else:
                approx += comp_temp
        
        return approx


# Look into optional normalization when you decide to make it a general library
def DeComp(T, rank=1, lr=1e-3, optimizier='ADAM', criterion=torch.nn.SmoothL1Loss(reduction='sum'), thresh_error=1e-9, max_epoch=1e4, patience_fac=1e-9, patience_perc=1, normalize=True, leave=True, cmplx=False):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)
    T = T.to(device)
    shape_tuple = T.shape

    my_cores = Cores(shape_tuple, rank, cmplx).to(device)
    param_list = []
    for name, param in my_cores.named_parameters():
        if name[-2:] == "_v":
            param_list.append(param)

    error = float("inf")
    error_list = []    
    patience_idx = int(patience_perc*max_epoch/100)

    if optimizier == 'SGD':
        optimizier = torch.optim.SGD(param_list, lr)
    if optimizier == 'ADAM':
        optimizier = torch.optim.Adam(param_list, lr)
    

    epoch_tq = tqdm(range(int(max_epoch)), leave=leave)
    for epoch in epoch_tq:
        epoch_tq.set_description(f"Loss: {error}")
        optimizier.zero_grad()

        approx = my_cores(device)

        loss = criterion(approx, T)
        error = loss.item()
        error_list.append(error)

        # Probably keep "initial patience kick in" as a different variable
        if epoch > patience_idx:
            mean_error = np.mean(np.array(error_list[-patience_idx:]))
            max_deviation = np.max(np.abs(np.array(error_list[-patience_idx:]) - mean_error))
            if max_deviation < patience_fac * mean_error:
                print(f"\nError Stagnant at: {error}\n")
                out_list = []
                for module in my_cores.normalized_list:
                    module = module.cpu()
                    temp = module(torch.LongTensor([0])).view(-1, 1)
                    for r in range(1, my_cores.rank):
                        temp = torch.cat((temp, module(torch.LongTensor([r])).view(-1, 1)), dim=1)
                    out_list.append(temp)
                return out_list



        if error < thresh_error:
            print(f"\nConverged with error: {error}\n")
            out_list = []
            for module in my_cores.normalized_list:
                module = module.cpu()
                temp = module(torch.LongTensor([0])).view(-1, 1)
                for r in range(1, my_cores.rank):
                    temp = torch.cat( (temp, module(torch.LongTensor([r])).view(-1, 1) ), dim=1)
                out_list.append(temp)
            return out_list

        loss.backward()
        optimizier.step()

    print(f"\nMaximum Epochs reached with error: {error}\n")
    out_list = []
    for module in my_cores.normalized_list:
        module = module.cpu()
        temp = module(torch.LongTensor([0])).view(-1, 1)
        for r in range(1, my_cores.rank):
            temp = torch.cat((temp, module(torch.LongTensor([r])).view(-1, 1)), dim=1)
        out_list.append(temp)
    return out_list


def sym_pow_decomp(T, criterion=torch.nn.SmoothL1Loss(reduction='sum'), thresh_error=1e-9, max_epoch=1e4, initialization=torch.randn, patience_fac=1e-9, patience_perc=1, normalize=True, leave=True, cmplx=False):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    T = T.to(device)

    shape_tuple = tuple(T.shape)
    assert shape_tuple == (shape_tuple[0],)*len(shape_tuple)

    num_legs = len(shape_tuple)
    leg_dim = shape_tuple[0]

    a = initialization(1, leg_dim).to(device)

    if cmplx:
        a = a.to(torch.cfloat)
        a += 1j*initialization(1, leg_dim).to(device)
        a /= math.sqrt(2)

    error = float("inf")
    error_list = []    
    patience_idx = int(patience_perc*max_epoch/100)

    epoch_tq = tqdm(range(int(max_epoch)), leave=leave)
    for epoch in epoch_tq:
        epoch_tq.set_description(f"Loss: {error}")

        view_T = T.view(-1, shape_tuple[0])

        '''
        Discussion part starts here
        ==================================================================
        '''
        if cmplx:
            norm_factor = torch.sqrt(torch.abs(a.conj().view(1, -1)@a.view(-1, 1))) ** (num_legs-1)
            # assert np.isclose(norm_factor.cpu(), 1.0)
        else:
            norm_factor = torch.norm(a) ** (num_legs-1)
            # norm_factor = (a.view(1,-1)@a.view(-1, 1)) ** (num_legs-1)
        
        # a1 = view_T @ CandyComp([a]*(num_legs-1)).view(-1, 1) / norm_factor
        # a2 = view_T @ CandyComp([a]*(num_legs-1)).view(-1, 1)
        # assert np.isclose(a1.cpu(), a2.cpu()).all()
        # a = a2
        a = CandyComp([a]*(num_legs-1)).view(1, -1) @ view_T / norm_factor
        if cmplx:
            a /= torch.abs(torch.sqrt(a.conj().view(1, -1)@a.view(-1, 1)))
            assert np.isclose(torch.abs(torch.sqrt(a.conj().view(1, -1)@a.view(-1, 1))).cpu(), 1.0)
        else:
            a /= torch.norm(a)
        '''
        ==================================================================
        Discussion part ends here
        '''

        approx = CandyComp([a]*num_legs)

        if cmplx:
            loss = torch.abs(torch.sum((approx - T)*(approx - T).conj()))
        else:
            loss = criterion(approx, T)
        error = loss.item()
        error_list.append(error)

        if epoch > patience_idx:
            mean_error = np.mean(np.array(error_list[-patience_idx:]))
            max_deviation = np.max(np.abs(np.array(error_list[-patience_idx:]) - mean_error))
            if max_deviation < patience_fac * mean_error:
                print(f"\nError Stagnant at: {error}\n")
                return a.cpu()

    
        if error < thresh_error:
            print(f"\nConverged with error: {error}\n")
            
            return a.cpu()
        
    print(f"\nMaximum Epochs reached with error: {error}\n")
    
    return a.cpu()


def SymDeComp(T, rank=1, lr=1e-3, optimizier='ADAM', criterion=torch.nn.SmoothL1Loss(reduction='sum'), thresh_error=1e-9, max_epoch=1e4, initialization=torch.randn, patience_fac=1e-9, patience_perc=1, normalize=True, leave=True):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    T = T.to(device)
    shape_tuple = T.shape

    my_cores = SymCores(shape_tuple, rank).to(device)
    param_list = []
    for name, param in my_cores.named_parameters():
        if name[-2:] == "_v":
            param_list.append(param)

    error = float("inf")
    error_list = []    
    patience_idx = int(patience_perc*max_epoch/100)

    if optimizier == 'SGD':
        optimizier = torch.optim.SGD(param_list, lr)
    if optimizier == 'ADAM':
        optimizier = torch.optim.Adam(param_list, lr)
    # Not a good idea to use ADAMW here, because you do not want to mess with the weights apart from the initial normalization
    # if optimizier == 'ADAMW':
    #     optimizier = torch.optim.AdamW(param_list, lr)
    

    epoch_tq = tqdm(range(int(max_epoch)), leave=leave)
    for epoch in epoch_tq:
        epoch_tq.set_description(f"Loss: {error}")
        optimizier.zero_grad()

        approx = my_cores(device)

        loss = criterion(approx, T)
        error = loss.item()
        error_list.append(error)

        if epoch > patience_idx:
            mean_error = np.mean(np.array(error_list[-patience_idx:]))
            max_deviation = np.max(np.abs(np.array(error_list[-patience_idx:]) - mean_error))
            if max_deviation < patience_fac * mean_error:
                print(f"\nError Stagnant at: {error}\n")
                out_list = []
                module = my_cores.m.cpu()
                temp = module(torch.LongTensor([0])).view(-1, 1)
                for r in range(1, my_cores.rank):
                    temp = torch.cat((temp, module(torch.LongTensor([r])).view(-1, 1)), dim=1)
                out_list.append(temp)
                return out_list



        if error < thresh_error:
            print(f"\nConverged with error: {error}\n")
            out_list = []
            module = my_cores.m.cpu()
            temp = module(torch.LongTensor([0])).view(-1, 1)
            for r in range(1, my_cores.rank):
                temp = torch.cat((temp, module(torch.LongTensor([r])).view(-1, 1)), dim=1)
            out_list.append(temp)
            return out_list

        loss.backward()
        optimizier.step()
    
    print(f"\nMaximum Epochs reached with error: {error}\n")
    out_list = []
    module = my_cores.m.cpu()
    temp = module(torch.LongTensor([0])).view(-1, 1)
    for r in range(1, my_cores.rank):
        temp = torch.cat((temp, module(torch.LongTensor([r])).view(-1, 1)), dim=1)
    out_list.append(temp)
    return out_list

#endregion



if __name__ == '__main__':
    # dim = 10
    # order = 4
    # batched = True
    # batch_size = 1
    # a = make_Gauss(dim, order, cmplx=True, batched=batched, batch_size=batch_size)
    # print(get_size(a))
    # b = torch.randn(2, 3, 4, device='cuda')
    # c = Tensaur(b)
    # # print(c.data_.device)
    # a = CompSetList([torch.randn(4, 2, 100)+1j*torch.randn(4, 2, 100), torch.randn(4, 2, 200)+1j*torch.randn(4, 2, 200), torch.randn(4, 2, 300)+1j*torch.randn(4, 2, 300)])
    # a_n = normalize_compsetlist(a)
    # print(torch.norm(a_n.data_[0], dim=-1))

    # a = make_GaussMPS(dim_list=[16, 17, 18, 19], bond_dim_list=[2, 3, 4], cyclic=False, batched=True, batch_size=32)
    # print(a.shape)

    # comp_set_list = [torch.randn(16, 2, 4), torch.randn(16, 2, 5), torch.randn(16, 2, 6)]
    # comp_set_list = normalize_compsetlist(comp_set_list)
    # print(torch.norm(comp_set_list.data_[1][10, 0]))

    # a = torch.tensor([[[0, 0], [1., 0]], [[0, 1.], [0, 0]]])/math.sqrt(2)
    # print(par_symmetrize(a).data_)
    # print(symmetrize(a).data_)
    # print(1/3/math.sqrt(2))

    csv_file = open("mps_params_6.csv", "w")
    writer = csv.writer(csv_file)
    writer.writerow(["Periodic Boundary Coniditons", "Translational Invariance", "Complex", "C1", "C2", "C3", "C4", "C5", "C6"])
    for periodic in [True, False]:
        for rep in [True, False]:
            for cmplx in [True, False]:
                params = mps_fitter(f"Logs/GaussianMPS_[3, 2]_periodic_{periodic}_rep_{rep}_cmplx_{cmplx}_normalize_False_DF.json", order_list=[3])
                writer.writerow([periodic, rep, cmplx, params[0], params[1], params[2], params[3], params[4], params[5]])
                print(f"Periodic: {'{0: <5}'.format(str(periodic))}, Repeat: {'{0: <5}'.format(str(rep))}, Complex: {'{0: <5}'.format(str(cmplx))}, Paramters: {params[0]:.8f} , {params[1]:.8f}, {params[2]:.8f}, {params[3]:.8f}, {params[4]:.8f}, {params[5]:.8f}")
    csv_file.close()

    # csv_file = open("gauss_params_2.csv", "w")
    # writer = csv.writer(csv_file)
    # writer.writerow(["Normalize", "Complex", "Symmetrized", "Order", "C1", "C2"])
    # for normalize in [True, False]:
    #     for cmplx in [False, True]:
    #         for sym in [True, False]:
    #             for order in [2, 3]:
    #                 params = gauss_fitter(f"Logs\Gaussian_[3, 2]_range(250, 0, -5)_250_{cmplx}_{normalize}.json", order=order, sym=sym)
    #                 writer.writerow([normalize, cmplx, sym, order, params[0], params[1]])
    #                 print(f"Normalize: {'{0: <5}'.format(str(normalize))}, Complex: {'{0: <5}'.format(str(cmplx))}, Symmetrize: {'{0: <5}'.format(str(sym))}, Order: {order:.0f}, Paramters: {params[0]:.8f} , {params[1]:.8f}")
    
    csv_file.close()



