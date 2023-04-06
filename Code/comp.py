import math
import numpy as np
import random
import torch

from itertools import permutations
from sympy.combinatorics.permutations import Permutation
from wrapper import CompSetList, Tensaur


"""
============================================
Constructor Functions
============================================
"""
#region


def make_Dicke(partition_tuple, num_particles):
    '''
    partition_tuple: (k1, k2, .. kd)
    num_particles: order = k1 + k2 +...+ kd
    '''
    assert num_particles == torch.tensor(partition_tuple).sum().item()

    leg_dim = len(partition_tuple)

    unperm_list = []
    for i in range(leg_dim):
        unperm_list += [i for _ in range(partition_tuple[i])]

    auxiliary_perm_list = list(permutations(unperm_list))
    perm_list = []

    # Will need to change the if condition if the number of parties (d) changes. Currently it is written only for d=2
    # i.e. currently only k1 and k2 are considered. So 2 possible states 0 and 1. Hence the sum thing works.
    for perm in auxiliary_perm_list:
        if np.array(perm).sum() == partition_tuple[1] and perm not in perm_list:
            perm_list.append(perm)

    state = torch.zeros((leg_dim,)*num_particles)

    for perm in perm_list:
        state[perm] += 1

    numerator = math.factorial(num_particles)
    denominator = 1

    for part in partition_tuple:
        denominator *= math.factorial(part)

    norm_factor = math.sqrt(numerator / denominator)
    state /= norm_factor

    return state


def make_Antisym(dim, order=None):
    '''
    dim: d
    order: n
    '''
    if order is None:
        order = dim
    
    perm_list = list(permutations(range(order)))

    norm_factor = math.sqrt(math.factorial(order))

    state = torch.zeros((dim,)*order)

    for perm in perm_list:
        state[perm] += Permutation(perm).signature() * 1.0
    
    state /= norm_factor
    
    return state


def make_Gauss(dim, order=None, cmplx=False, batched=False, batch_size=None, mean=0, stdv=None):
    '''
    dim: d
    order: n
    '''

    if batched ^ bool(batch_size):
        raise ValueError(f"Expected (batched, batch_size) to be (True, int), (False, None) or (False, 0) but got ({batched}, {batch_size}).")
    
    if order is None:
        order = dim

    shape_tuple = (dim,)*order
    if batched:
        shape_tuple = (batch_size,) + shape_tuple

    if stdv is None:
        stdv = math.sqrt(2/dim)

    out = torch.normal(mean, stdv, shape_tuple)
    if cmplx:
        comp_init = 1j*torch.normal(mean, stdv, shape_tuple)
        out = (out + comp_init) / math.sqrt(2)

    return Tensaur(out, batched=batched)


def make_SymGauss(dim, order=None, cmplx=False, batched=False, batch_size=None, mean=0, stdv=None):
    '''
    dim: d
    order: n
    batch_size: b
    '''
    if batched ^ bool(batch_size):
        raise ValueError(f"Expected (batched, batch_size) to be (True, int), (False, None) or (False, 0) but got ({batched}, {batch_size}).")
    if order is None:
        order = dim
    
    shape_tuple = (dim,)*order
    if batched:
        shape_tuple = (batch_size,) + shape_tuple

    if stdv is None:
        stdv = math.sqrt(2/dim)

    initial = torch.normal(mean, stdv, shape_tuple)
    if cmplx:
        comp_init = 1j*torch.normal(mean, stdv, shape_tuple)
        initial = (initial + comp_init) / math.sqrt(2)

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
            out = initial.clone()
        else:
            out += initial.clone().permute(perm)
            
    assert norm_factor == count
    out /= norm_factor

    return Tensaur(out, batched=batched)


def make_GaussMPS_general(dim_list, bond_dim_list=None, cyclic=False, cmplx=False, batched=False, batch_size=None, mean_list=None, stdv_list=None):
    '''
    dim: d
    bond_dim = d_bond
    num_sites: n
    batch_size: b
    '''

    if batched ^ bool(batch_size):
        raise ValueError(f"Expected (batched, batch_size) to be (True, int), (False, None) or (False, 0) but got ({batched}, {batch_size}).")
    
    num_sites = len(dim_list)

    if not cyclic:
        for idx in range(0, num_sites):
            if idx == 0:
                temp_shape_tuple = (dim_list[idx], bond_dim_list[idx])
            elif idx == num_sites-1 :
                temp_shape_tuple = (bond_dim_list[idx-1], dim_list[idx])
            else:
                temp_shape_tuple = (bond_dim_list[idx-1], dim_list[idx], bond_dim_list[idx])

            if batched:
                temp_shape_tuple = (batch_size,) + temp_shape_tuple
        
            if mean_list is None:
                mean = 0
            else:
                mean = mean_list[idx]
        
            if stdv_list is None:
                stdv = math.sqrt(1/dim_list[idx])
            else:
                stdv = stdv_list[idx]

            site_temp = torch.normal(mean, stdv, temp_shape_tuple)

            if cmplx:
                site_temp_comp = torch.normal(mean, stdv, temp_shape_tuple)
                site_temp = (site_temp + 1j*site_temp_comp) / math.sqrt(2)
            
            if batched:
                if idx == 0:
                    out = site_temp
                elif idx == num_sites - 1:
                    out = torch.einsum('b...ij, bjk -> b...ik', out, site_temp)
                else:
                    out = torch.einsum('b...ij, bjkl -> b...ikl', out, site_temp)

            else:
                if idx == 0:
                    out = site_temp
                elif idx == num_sites - 1:
                    out = torch.einsum('...ij, jk -> ...ik', out, site_temp)
                else:
                    out = torch.einsum('...ij, jkl -> ...ikl', out, site_temp)

    # else:
    #     for idx in range(0, num_sites):

    #         temp_shape_tuple = (bond_dim_list[idx-1], dim_list[idx], bond_dim_list[idx])
        
    #         if batched:
    #                 temp_shape_tuple = (batch_size,) + temp_shape_tuple
            
    #         if mean_list is None:
    #             mean = 0
    #         else:
    #             mean = mean_list[idx]
        
    #         if stdv_list is None:
    #             stdv = math.sqrt(1/dim_list[idx])
    #         else:
    #             stdv = stdv_list[idx]

    #         site_temp = torch.normal(mean, stdv, temp_shape_tuple)

    #         if cmplx:
    #             site_temp_comp = torch.normal(mean, stdv, shape)
    #             site_temp = (site_temp + site_temp_comp) / math.sqrt(2)

    #         if idx == 0:
    #             site_zero = site_temp
    #         if idx == num_sites - 1:
    #             site_last = site_temp
            
    #         if batched:
    #             if idx == 0:
    #                 out = site_temp
    #             elif idx == num_sites - 1:
    #                 out = torch.einsum('b...ij, bjk -> b...ik', out, site_temp)
    #             else:
    #                 out = torch.einsum('b...ij, bjkl -> b...ikl', out, site_temp)

    #         else:
    #             if idx == 0:
    #                 out = site_temp
    #             elif idx == num_sites - 1:
    #                 out = torch.einsum('...ij, jk -> ...ik', out, site_temp)
    #             else:
    #                 out = torch.einsum('...ij, jkl -> ...ikl', out, site_temp)

    return Tensaur(out, batched=batched)
    

def make_GaussMPS_special(num_sites, bond_dim, dim, periodic=False, rep=False, cmplx=False, batched=False, batch_size=None, mean=None, stdv=None):
    '''
    dim: d
    bond_dim = d_bond
    num_sites: n
    batch_size: b
    '''
    assert batch_size is not None
    assert batched == True

    if batched ^ bool(batch_size):
        raise ValueError(f"Expected (batched, batch_size) to be (True, int), (False, None) or (False, 0) but got ({batched}, {batch_size}).")
    
    if mean is None:
        mean = 0
    if stdv is None:
        stdv = math.sqrt(1/(dim*bond_dim))

    if periodic:
        if rep:
            site_temp = torch.normal(mean, stdv, (batch_size, bond_dim, dim, bond_dim))
            if cmplx:
                site_temp = (site_temp + 1j*torch.normal(mean, stdv, (batch_size, bond_dim, dim, bond_dim)))/math.sqrt(2)
            site_sample_list = [site_temp]*num_sites

        else:
            site_sample_list = [torch.normal(mean, stdv, (batch_size, bond_dim, dim, bond_dim)) for _ in range(num_sites)]
            if cmplx:
                site_sample_list = [(x + 1j*torch.normal(mean, stdv, x.shape))/math.sqrt(2) for x in site_sample_list]

        out = site_sample_list[0]
        for idx in range(1, num_sites):
            out = torch.einsum('b...ij, bjkl -> b...ikl', out, site_sample_list[idx])
        
        out = torch.einsum('bi...i -> b...', out)

    else:
        if rep:
            site_temp_start = torch.normal(mean, stdv, (batch_size, dim, bond_dim))
            site_temp_mid = torch.normal(mean, stdv, (batch_size, bond_dim, dim, bond_dim))
            site_temp_end = torch.normal(mean, stdv, (batch_size, bond_dim, dim))
            if cmplx:
                site_temp_start = (site_temp_start + 1j*torch.normal(mean, stdv, (batch_size, dim, bond_dim)))/math.sqrt(2)
                site_temp_mid = (site_temp_mid + 1j*torch.normal(mean, stdv, (batch_size, bond_dim, dim, bond_dim)))/math.sqrt(2)
                site_temp_end = (site_temp_end + 1j*torch.normal(mean, stdv, (batch_size, bond_dim, dim)))/ math.sqrt(2)
        
            site_sample_list = [site_temp_start] + [site_temp_mid]*(num_sites-2) + [site_temp_end]
        

        else:
            site_sample_list = [torch.normal(mean, stdv, (batch_size, dim, bond_dim))] + [torch.normal(mean, stdv, (batch_size, bond_dim, dim, bond_dim)) for _ in range(num_sites-2)] + [torch.normal(mean, stdv, (batch_size, bond_dim, dim))]
            if cmplx:
                site_sample_list = [(x + 1j*torch.normal(mean, stdv, x.shape))/math.sqrt(2) for x in site_sample_list]
        
        out = site_sample_list[0]
        for idx in range(1, num_sites-1):
            out = torch.einsum('b...ij, bjkl -> b...ikl', out, site_sample_list[idx])
        out = torch.einsum('b...ij, bjk -> b...ik', out, site_sample_list[-1])
    
    return Tensaur(out, batched=batched)
#endregion



def CandyComp(comp_set_list, batched=False):
    
    # Typecasting
    if not isinstance(comp_set_list, CompSetList):
        comp_set_list = CompSetList(comp_set_list)

    # Initialize
    comp_out = comp_set_list.data_[0]

    # Outerproduct
    for comp_set in comp_set_list.data_[1:]:
        # print("Comp set shape: ", comp_set.shape)
        if comp_set_list.batched:
            comp_out = torch.einsum("br...ij, br...jk -> br...ik", comp_out.unsqueeze(-1), comp_set.unsqueeze(-2))
        else:
            comp_out = torch.einsum("r...ij, r...jk -> r...ik", comp_out.unsqueeze(-1), comp_set.unsqueeze(-2))
        # print("Comp out shape: ", comp_out.shape)
    # Summation
    if comp_set_list.batched:
        return comp_out.sum(dim=1)
    else:
        return comp_out.sum(dim=0)



if __name__ == '__main__':
    a = make_GaussMPS_special(3, 10, 25, periodic=False, rep=False, cmplx=False, batched=True, batch_size=16)
    print("printing a shape", a.data_.shape)
    print(a.data_.dtype)