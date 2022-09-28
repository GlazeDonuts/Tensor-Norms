# Imports
import comp
import json
import math
import numpy as np
import sys
import time
import torch
import tntorch as tn
import utils

from config import def_config
from tqdm import tqdm
from comp import CandyComp
from decomp import DeComp, SymDeComp, SymPowDeComp
from wrapper import Tensaur, CompSetList


def GaussExp(   dim_list,   order_list,     num_samples,    batch_size=None,
                alloc=3.2,  cmplx=False,    write_freq=1,   write_path=None,
                device=None, config=def_config,   normalize=False):
    '''
    Gaussian State experiment
    dim_list: values of d (>> values of n)
    order_list: values of n
    num_samples: number of samples for each (order, dim)
    batch_size: size of each batch < number of samples
    cmplx: bool for complex or real
    normalize: bool for normalizing tensors pre-optimzation
    '''

    utils.set_seed(config.seed)

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    if write_path is None:
        write_path = f"Logs/Gaussian_{str(order_list)}_{str(dim_list)}_{num_samples}_{str(cmplx)}_{str(normalize)}.json"
    
    logger = {}
    logger['exp_start_time'] = time.strftime('%X %x %Z')
    logger['alloc'] = alloc
    logger['complex'] = cmplx
    logger['order_list'] = []

    for order in order_list:
        logger[str(order)] = {}
        logger[str(order)]['dim_list'] = []
        if cmplx == False:
            logger[str(order)]['avg_non_als_gm_list'] = []
            logger[str(order)]['avg_par_als_gm_list'] = []
            logger[str(order)]['avg_sym_als_gm_list'] = []
        logger[str(order)]['avg_non_ngd_gm_list'] = []
        logger[str(order)]['avg_par_ngd_gm_list'] = []
        logger[str(order)]['avg_sym_ngd_gm_list'] = []
        logger[str(order)]['avg_sym_sgd_gm_list'] = []
        logger[str(order)]['avg_sym_pow_gm_list'] = []               
        
        print("="*100)

        for dim_idx, dim in enumerate(dim_list):
            logger[str(order)][str(dim)] = {}

            if batch_size is None:
                temp = comp.make_Gauss(dim=dim, order=order, cmplx=cmplx)
                temp_space = DeComp(temp, decomp_only_init=True)
                batch_size_list = utils.make_batch_list(temp_space, num_samples=num_samples, alloc=alloc, cmplx=cmplx)
                del temp
                del temp_space
                torch.cuda.empty_cache()
            else:
                num_reg_batches = num_samples//batch_size
                last_batch_size = num_samples - num_reg_batches*batch_size
                if last_batch_size == 0:
                    batch_size_list = [int(batch_size)]*num_reg_batches
                else:
                    batch_size_list = [int(batch_size)]*num_reg_batches + [int(last_batch_size)]

            logger[str(order)][str(dim)]['batch_size_list'] = batch_size_list
            logger[str(order)][str(dim)]['start_time'] = time.strftime('%X %x %Z')
            print(time.strftime('%X %x %Z'))
            print(f"Order: {order}, Dimension: {dim}, Complex: {cmplx}, Normalize: {normalize}")

            logger[str(order)][str(dim)]['non_l2_norm_list'] = []
            logger[str(order)][str(dim)]['par_l2_norm_list'] = []
            logger[str(order)][str(dim)]['sym_l2_norm_list'] = []
            if cmplx == False:
                logger[str(order)][str(dim)]['non_als_gm_list'] = []
                logger[str(order)][str(dim)]['par_als_gm_list'] = []
                logger[str(order)][str(dim)]['sym_als_gm_list'] = []
            logger[str(order)][str(dim)]['non_ngd_gm_list'] = []
            logger[str(order)][str(dim)]['par_ngd_gm_list'] = []
            logger[str(order)][str(dim)]['sym_ngd_gm_list'] = []
            logger[str(order)][str(dim)]['sym_sgd_gm_list'] = []
            logger[str(order)][str(dim)]['sym_pow_gm_list'] = []
            
            logger[str(order)][str(dim)]['non_ngd_loss_list'] = []
            logger[str(order)][str(dim)]['par_ngd_loss_list'] = []
            logger[str(order)][str(dim)]['sym_ngd_loss_list'] = []
            logger[str(order)][str(dim)]['sym_sgd_loss_list'] = []
            logger[str(order)][str(dim)]['sym_pow_loss_list'] = []
            

            for batch_idx, curr_batch_size in enumerate(batch_size_list):
                print("="*75)
                print(f"Batch: {batch_idx+1} of {len(batch_size_list)}, Size: {curr_batch_size}")

                # Non-Symmetrized State
                #region
                print("============ Non-Symmetrized ============")
                non_state_batch = comp.make_Gauss(dim, order, batched=True, batch_size=curr_batch_size, cmplx=cmplx)
                temp_non_state_batch = non_state_batch
                for _ in range(curr_batch_size):
                    logger[str(order)][str(dim)]['non_l2_norm_list'].append(torch.norm(non_state_batch.data_[_]).item())
                    if normalize:
                        non_state_batch.data_[_] /= torch.norm(non_state_batch.data_[_])
                        assert torch.isclose(torch.norm(non_state_batch.data_[_]), torch.tensor(1.0), rtol=config.rtol).item() == True

                if cmplx == False:
                    print("Starting ALS")
                    non_als_cores = tn.Tensor(non_state_batch.data_, batch=True, ranks_cp=1).cores
                    non_als_cores = [x.permute([0] + [x for x in range(1,len(x.shape))[::-1]]) for x in non_als_cores]
                    non_als_approx = CandyComp(non_als_cores)
                    for _ in range(curr_batch_size):
                        non_als_approx[_] /= torch.norm(non_als_approx[_])
                        assert torch.isclose(torch.norm(non_als_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    logger[str(order)][str(dim)]['non_als_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', non_als_approx.view(curr_batch_size, -1), non_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())

                non_ngd_cores, non_ngd_loss = DeComp(non_state_batch, batched=True, config=config)
                non_ngd_approx = CandyComp(non_ngd_cores)
                for _ in range(curr_batch_size):
                    non_ngd_approx[_] /= torch.norm(non_ngd_approx[_])
                    assert torch.isclose(torch.norm(non_ngd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True

                logger[str(order)][str(dim)]['non_ngd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', non_ngd_approx.view(curr_batch_size, -1), non_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                logger[str(order)][str(dim)]['non_ngd_loss_list'].append(non_ngd_loss)
                
                del non_state_batch
                if cmplx == False:
                    del non_als_cores
                del non_ngd_cores
                del non_ngd_loss
                del non_ngd_approx
                torch.cuda.empty_cache()
                #endregion


                # Partialy Symmetrized
                #region
                print("========= Partially Symmetrized =========")
                par_state_batch = utils.par_symmetrize(temp_non_state_batch, batched=True)
                for _ in range(curr_batch_size):
                    logger[str(order)][str(dim)]['par_l2_norm_list'].append(torch.norm(par_state_batch.data_[_]).item())
                    if normalize:
                        par_state_batch.data_[_] /= torch.norm(par_state_batch.data_[_])
                        assert torch.isclose(torch.norm(par_state_batch.data_[_]), torch.tensor(1.0), rtol=config.rtol).item() == True

                if cmplx == False:
                    print("Starting ALS")
                    par_als_cores = tn.Tensor(par_state_batch.data_, batch=True, ranks_cp=1).cores
                    par_als_cores = [x.permute([0] + [x for x in range(1,len(x.shape))[::-1]]) for x in par_als_cores]
                    par_als_approx = CandyComp(par_als_cores)
                    for _ in range(curr_batch_size):
                        par_als_approx[_] /= torch.norm(par_als_approx[_])
                        assert torch.isclose(torch.norm(par_als_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    logger[str(order)][str(dim)]['par_als_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', par_als_approx.view(curr_batch_size, -1), par_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())

                par_ngd_cores, par_ngd_loss = DeComp(par_state_batch, batched=True, config=config)
                par_ngd_approx = CandyComp(par_ngd_cores)
                for _ in range(curr_batch_size):
                    par_ngd_approx[_] /= torch.norm(par_ngd_approx[_])
                    assert torch.isclose(torch.norm(par_ngd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True

                logger[str(order)][str(dim)]['par_ngd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', par_ngd_approx.view(curr_batch_size, -1), par_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                logger[str(order)][str(dim)]['par_ngd_loss_list'].append(par_ngd_loss)
                
                del par_state_batch
                if cmplx == False:
                    del par_als_cores
                del par_ngd_cores
                del par_ngd_loss
                del par_ngd_approx
                torch.cuda.empty_cache()
                #endregion

                
                # Fully Symmetrized
                #region
                print("=========== Fully Symmetrized ===========")
                sym_state_batch = utils.symmetrize(temp_non_state_batch, batched=True)
                del temp_non_state_batch
                for _ in range(curr_batch_size):
                    logger[str(order)][str(dim)]['sym_l2_norm_list'].append(torch.norm(sym_state_batch.data_[_]).item())
                    if normalize:
                        sym_state_batch.data_[_] /= torch.norm(sym_state_batch.data_[_])
                        assert torch.isclose(torch.norm(sym_state_batch.data_[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                
                if cmplx == False:
                    print("Starting ALS")
                    sym_als_cores = tn.Tensor(sym_state_batch.data_, batch=True, ranks_cp=1).cores
                    sym_als_cores = [x.permute([0] + [x for x in range(1,len(x.shape))[::-1]]) for x in sym_als_cores]
                    sym_als_approx = CandyComp(sym_als_cores)
                    for _ in range(curr_batch_size):
                        sym_als_approx[_] /= torch.norm(sym_als_approx[_])
                        assert torch.isclose(torch.norm(sym_als_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True                    
                    logger[str(order)][str(dim)]['sym_als_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', sym_als_approx.view(curr_batch_size, -1), sym_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                
                sym_ngd_cores, sym_ngd_loss = DeComp(sym_state_batch, batched=True, config=config)
                sym_sgd_cores, sym_sgd_loss = SymDeComp(sym_state_batch, batched=True, config=config)
                sym_pow_cores, sym_pow_loss = SymPowDeComp(sym_state_batch, batched=True, config=config)

                sym_ngd_approx = CandyComp(sym_ngd_cores)
                sym_sgd_approx = CandyComp(sym_sgd_cores)
                sym_pow_approx = CandyComp(sym_pow_cores)

                for _ in range(curr_batch_size):
                    sym_ngd_approx[_] /= torch.norm(sym_ngd_approx[_])
                    sym_sgd_approx[_] /= torch.norm(sym_sgd_approx[_])
                    sym_pow_approx[_] /= torch.norm(sym_pow_approx[_])
                    assert torch.isclose(torch.norm(sym_ngd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    assert torch.isclose(torch.norm(sym_sgd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    assert torch.isclose(torch.norm(sym_pow_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True

                
                logger[str(order)][str(dim)]['sym_ngd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', sym_ngd_approx.view(curr_batch_size, -1), sym_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                logger[str(order)][str(dim)]['sym_sgd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', sym_sgd_approx.view(curr_batch_size, -1), sym_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                logger[str(order)][str(dim)]['sym_pow_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', sym_pow_approx.view(curr_batch_size, -1), sym_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())

                logger[str(order)][str(dim)]['sym_ngd_loss_list'].append(sym_ngd_loss)
                logger[str(order)][str(dim)]['sym_sgd_loss_list'].append(sym_sgd_loss)
                logger[str(order)][str(dim)]['sym_pow_loss_list'].append(sym_pow_loss)

                del temp_non_state_batch
                del sym_state_batch
                if cmplx == False:
                    del sym_als_cores
                    del non_als_approx
                    del sym_als_approx
                del sym_ngd_cores
                del sym_sgd_cores
                del sym_pow_cores
                del sym_ngd_loss
                del sym_sgd_loss
                del sym_pow_loss
                del sym_ngd_approx
                del sym_sgd_approx
                del sym_pow_approx
                torch.cuda.empty_cache()
                #endregion

            if cmplx == False:
                logger[str(order)]['avg_non_als_gm_list'].append(float(np.mean(logger[str(order)][str(dim)]['non_als_gm_list'])))
                logger[str(order)]['avg_par_als_gm_list'].append(float(np.mean(logger[str(order)][str(dim)]['par_als_gm_list'])))
                logger[str(order)]['avg_sym_als_gm_list'].append(float(np.mean(logger[str(order)][str(dim)]['sym_als_gm_list'])))
            logger[str(order)]['avg_non_ngd_gm_list'].append(float(np.mean(logger[str(order)][str(dim)]['non_ngd_gm_list'])))
            logger[str(order)]['avg_par_ngd_gm_list'].append(float(np.mean(logger[str(order)][str(dim)]['par_ngd_gm_list'])))
            logger[str(order)]['avg_sym_ngd_gm_list'].append(float(np.mean(logger[str(order)][str(dim)]['sym_ngd_gm_list'])))
            logger[str(order)]['avg_sym_sgd_gm_list'].append(float(np.mean(logger[str(order)][str(dim)]['sym_sgd_gm_list'])))
            logger[str(order)]['avg_sym_pow_gm_list'].append(float(np.mean(logger[str(order)][str(dim)]['sym_pow_gm_list'])))

            logger[str(order)]['dim_list'].append(dim)
            if dim_idx % write_freq == 0:
                dump_object = json.dumps(logger)
                with open(write_path, "w") as outfile:
                    outfile.write(dump_object)

            logger[str(order)][str(dim)]['end_time'] = time.strftime('%X %x %Z')

        logger['order_list'].append(order)
        dump_object = json.dumps(logger)
        with open(write_path, "w") as outfile:
            outfile.write(dump_object)

    logger['exp_end_time'] = time.strftime('%X %x %Z')


def GaussMPSExp_DF(order_list, dim_list, bond_dim_list, num_samples, periodic, rep, cmplx=False, normalize=False, batch_size=None, alloc_gpu=3.2, alloc_cpu=14, write_freq=1, write_path=None, device=None, config=def_config):
    '''
    Gaussian MPS experiment
    order_list: values of n
    bond_dim_list: values of q
    dim_list: values of d
    num_samples: number of samples for each (n, q, d)
    batch_size: size of each batch <= number of samples
    '''
    utils.set_seed(config.seed)

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    if write_path is None:
        write_path = f"Logs/GaussianMPS_{str(order_list)}_periodic_{str(periodic)}_rep_{str(rep)}_cmplx_{str(cmplx)}_normalize_{str(normalize)}_DF.json"
    
    logger = {}
    logger['exp_start_time'] = time.strftime('%X %x %Z')
    logger['alloc_gpu'] = alloc_gpu
    logger['alloc_cpu'] = alloc_cpu
    logger['complex'] = cmplx
    logger['order_list'] = []

    for order in order_list:

        logger[str(order)] = {}
        logger[str(order)]['dim_list'] = []

        for dim in dim_list:

            logger[str(order)][str(dim)] = {}
            logger[str(order)][str(dim)]['bond_dim_list'] = []

            if cmplx == False:
                logger[str(order)][str(dim)]['avg_non_als_gm_list'] = []
                logger[str(order)][str(dim)]['avg_par_als_gm_list'] = []
                logger[str(order)][str(dim)]['avg_sym_als_gm_list'] = []

            logger[str(order)][str(dim)]['avg_non_ngd_gm_list'] = []
            logger[str(order)][str(dim)]['avg_par_ngd_gm_list'] = []
            logger[str(order)][str(dim)]['avg_sym_ngd_gm_list'] = []
            logger[str(order)][str(dim)]['avg_sym_sgd_gm_list'] = []
            logger[str(order)][str(dim)]['avg_sym_pow_gm_list'] = []
            
            for bond_dim_idx, bond_dim in enumerate(bond_dim_list):
                
                logger[str(order)][str(dim)][str(bond_dim)] = {}
                logger[str(order)][str(dim)][str(bond_dim)]['start_time'] = time.strftime('%X %x %Z')
                if batch_size is None:
                    temp = comp.make_GaussMPS_special(num_sites=order, bond_dim=bond_dim, dim=dim, cmplx=cmplx, batched=True, batch_size=1)
                    temp_space = DeComp(temp, decomp_only_init=True)
                    batch_size_list = utils.make_batch_list_mps(order, bond_dim, dim, temp_space, num_samples=num_samples, alloc_gpu=alloc_gpu, alloc_cpu=alloc_cpu, cmplx=cmplx)
                    del temp
                    del temp_space
                    torch.cuda.empty_cache()
                else:
                    num_reg_batches = num_samples//batch_size
                    last_batch_size = num_samples - num_reg_batches*batch_size
                    if last_batch_size == 0:
                        batch_size_list = [int(batch_size)]*num_reg_batches
                    else:
                        batch_size_list = [int(batch_size)]*num_reg_batches + [int(last_batch_size)]
                
                
                logger[str(order)][str(dim)][str(bond_dim)]['batch_size_list'] = batch_size_list
                print(time.strftime('%X %x %Z'))
                print(f"Order: {order}, Dimension: {dim}, Bond Dimension: {bond_dim}")
                print(f"Periodic: {periodic}, Repeat: {rep}, Complex: {cmplx}, Normalize: {normalize}")

                logger[str(order)][str(dim)][str(bond_dim)]['non_l2_norm_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['par_l2_norm_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['sym_l2_norm_list'] = []

                if cmplx == False:
                    logger[str(order)][str(dim)][str(bond_dim)]['non_als_gm_list'] = []
                    logger[str(order)][str(dim)][str(bond_dim)]['par_als_gm_list'] = []
                    logger[str(order)][str(dim)][str(bond_dim)]['sym_als_gm_list'] = []

                logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_gm_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['par_ngd_gm_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['sym_ngd_gm_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['sym_sgd_gm_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['sym_pow_gm_list'] = []

                logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_loss_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['par_ngd_loss_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['sym_ngd_loss_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['sym_sgd_loss_list'] = []
                logger[str(order)][str(dim)][str(bond_dim)]['sym_pow_loss_list'] = []

                for batch_idx, curr_batch_size in enumerate(batch_size_list):
                    print("="*75)
                    print(f"Batch: {batch_idx+1} of {len(batch_size_list)}, Size: {curr_batch_size}")


                    #region
                    print("============ Non-Symmetrized ============")
                    non_state_batch = comp.make_GaussMPS_special(num_sites=order, bond_dim=bond_dim, dim=dim, periodic=periodic, rep=rep, cmplx=cmplx, batched=True, batch_size=curr_batch_size)
                    temp_non_state_batch = non_state_batch
                    for _ in range(curr_batch_size):
                        logger[str(order)][str(dim)][str(bond_dim)]['non_l2_norm_list'].append(torch.norm(non_state_batch.data_[_]).item())
                        if normalize:
                            non_state_batch.data_[_] /= torch.norm(non_state_batch.data_[_])
                            assert torch.isclose(torch.norm(non_state_batch.data_[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    
                    if cmplx == False:
                        print("Starting ALS")
                        non_als_cores = tn.Tensor(non_state_batch.data_, batch=True, ranks_cp=1).cores
                        non_als_cores = [x.permute([0] + [x for x in range(1,len(x.shape))[::-1]]) for x in non_als_cores]
                        non_als_approx = CandyComp(non_als_cores)
                        
                        for _ in range(curr_batch_size):
                            non_als_approx[_] /= torch.norm(non_als_approx[_])
                            assert torch.isclose(torch.norm(non_als_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                        
                        logger[str(order)][str(dim)][str(bond_dim)]['non_als_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', non_als_approx.view(curr_batch_size, -1), non_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                    
                    non_ngd_cores, non_ngd_loss = DeComp(non_state_batch, batched=True, config=config)
                    non_ngd_approx = CandyComp(non_ngd_cores)

                    for _ in range(curr_batch_size):
                        assert torch.isclose(torch.norm(non_ngd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    
                    logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', non_ngd_approx.view(curr_batch_size, -1), non_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                    logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_loss_list'].append(non_ngd_loss)
                    
                    del non_state_batch
                    if cmplx == False:
                        del non_als_cores
                        del non_als_approx
                    del non_ngd_cores
                    del non_ngd_loss
                    del non_ngd_approx
                    torch.cuda.empty_cache()
                    #endregion


                    #region
                    print("========= Partially Symmetrized =========")
                    par_state_batch = utils.par_symmetrize(temp_non_state_batch, batched=True)
                    for _ in range(curr_batch_size):
                        logger[str(order)][str(dim)][str(bond_dim)]['par_l2_norm_list'].append(torch.norm(par_state_batch.data_[_]).item())
                        if normalize:
                            par_state_batch.data_[_] /= torch.norm(par_state_batch.data_[_])
                            assert torch.isclose(torch.norm(par_state_batch.data_[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    
                    if cmplx == False:
                        print("Starting ALS")
                        par_als_cores = tn.Tensor(par_state_batch.data_, batch=True, ranks_cp=1).cores
                        par_als_cores = [x.permute([0] + [x for x in range(1,len(x.shape))[::-1]]) for x in par_als_cores]
                        par_als_approx = CandyComp(par_als_cores)
                        
                        for _ in range(curr_batch_size):
                            par_als_approx[_] /= torch.norm(par_als_approx[_])
                            assert torch.isclose(torch.norm(par_als_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                        
                        logger[str(order)][str(dim)][str(bond_dim)]['par_als_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', par_als_approx.view(curr_batch_size, -1), par_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                    
                    par_ngd_cores, par_ngd_loss = DeComp(par_state_batch, batched=True, config=config)
                    par_ngd_approx = CandyComp(par_ngd_cores)

                    for _ in range(curr_batch_size):
                        assert torch.isclose(torch.norm(par_ngd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    
                    logger[str(order)][str(dim)][str(bond_dim)]['par_ngd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', par_ngd_approx.view(curr_batch_size, -1), par_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                    logger[str(order)][str(dim)][str(bond_dim)]['par_ngd_loss_list'].append(par_ngd_loss)
                    
                    del par_state_batch
                    if cmplx == False:
                        del par_als_cores
                        del par_als_approx
                    del par_ngd_cores
                    del par_ngd_loss
                    del par_ngd_approx
                    torch.cuda.empty_cache()
                    #endregion


                    #region
                    print("=========== Fully Symmetrized ===========")
                    sym_state_batch = utils.symmetrize(temp_non_state_batch, batched=True)
                    del temp_non_state_batch
                    for _ in range(curr_batch_size):
                        logger[str(order)][str(dim)][str(bond_dim)]['sym_l2_norm_list'].append(torch.norm(sym_state_batch.data_[_]).item())
                        if normalize:
                            sym_state_batch.data_[_] /= torch.norm(sym_state_batch.data_[_])
                            assert torch.isclose(torch.norm(sym_state_batch.data_[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                    
                    if cmplx == False:
                        print("Starting ALS")
                        sym_als_cores = tn.Tensor(sym_state_batch.data_, batch=True, ranks_cp=1).cores
                        sym_als_cores = [x.permute([0] + [x for x in range(1,len(x.shape))[::-1]]) for x in sym_als_cores]
                        sym_als_approx = CandyComp(sym_als_cores)
                
                        for _ in range(curr_batch_size):
                            sym_als_approx[_] /= torch.norm(sym_als_approx[_])
                            assert torch.isclose(torch.norm(sym_als_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                        
                        logger[str(order)][str(dim)][str(bond_dim)]['sym_als_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', sym_als_approx.view(curr_batch_size, -1), sym_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                    
                    sym_ngd_cores, sym_ngd_loss = DeComp(sym_state_batch, batched=True, config=config)
                    sym_sgd_cores, sym_sgd_loss = SymDeComp(sym_state_batch, batched=True, config=config)
                    sym_pow_cores, sym_pow_loss = SymPowDeComp(sym_state_batch, batched=True, config=config)
                    
                    sym_ngd_approx = CandyComp(sym_ngd_cores)
                    sym_sgd_approx = CandyComp(sym_sgd_cores)
                    sym_pow_approx = CandyComp(sym_pow_cores)

                    for _ in range(curr_batch_size):
                        assert torch.isclose(torch.norm(sym_ngd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                        assert torch.isclose(torch.norm(sym_sgd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
                        assert torch.isclose(torch.norm(sym_pow_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True

                    logger[str(order)][str(dim)][str(bond_dim)]['sym_ngd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', sym_ngd_approx.view(curr_batch_size, -1), sym_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                    logger[str(order)][str(dim)][str(bond_dim)]['sym_sgd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', sym_sgd_approx.view(curr_batch_size, -1), sym_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                    logger[str(order)][str(dim)][str(bond_dim)]['sym_pow_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', sym_pow_approx.view(curr_batch_size, -1), sym_state_batch.data_.view(curr_batch_size, -1).conj()).view(-1,)).tolist())
                
                    logger[str(order)][str(dim)][str(bond_dim)]['sym_ngd_loss_list'].append(sym_ngd_loss)
                    logger[str(order)][str(dim)][str(bond_dim)]['sym_sgd_loss_list'].append(sym_sgd_loss)
                    logger[str(order)][str(dim)][str(bond_dim)]['sym_pow_loss_list'].append(sym_pow_loss)

                    del sym_state_batch
                    if cmplx == False:
                        del sym_als_cores
                        del sym_als_approx
                    del sym_ngd_cores
                    del sym_sgd_cores
                    del sym_pow_cores
                    del sym_ngd_loss
                    del sym_sgd_loss
                    del sym_pow_loss
                    del sym_ngd_approx
                    del sym_sgd_approx
                    del sym_pow_approx
                    torch.cuda.empty_cache()
                    #endregion


                if cmplx == False:
                    logger[str(order)][str(dim)]['avg_non_als_gm_list'].append(float(np.mean(logger[str(order)][str(dim)][str(bond_dim)]['non_als_gm_list'])))
                    logger[str(order)][str(dim)]['avg_par_als_gm_list'].append(float(np.mean(logger[str(order)][str(dim)][str(bond_dim)]['par_als_gm_list'])))
                    logger[str(order)][str(dim)]['avg_sym_als_gm_list'].append(float(np.mean(logger[str(order)][str(dim)][str(bond_dim)]['sym_als_gm_list'])))
                
                logger[str(order)][str(dim)]['avg_non_ngd_gm_list'].append(float(np.mean(logger[str(order)][str(dim)][str(bond_dim)]['non_ngd_gm_list'])))
                logger[str(order)][str(dim)]['avg_par_ngd_gm_list'].append(float(np.mean(logger[str(order)][str(dim)][str(bond_dim)]['par_ngd_gm_list'])))
                logger[str(order)][str(dim)]['avg_sym_ngd_gm_list'].append(float(np.mean(logger[str(order)][str(dim)][str(bond_dim)]['sym_ngd_gm_list'])))
                logger[str(order)][str(dim)]['avg_sym_sgd_gm_list'].append(float(np.mean(logger[str(order)][str(dim)][str(bond_dim)]['sym_sgd_gm_list'])))
                logger[str(order)][str(dim)]['avg_sym_pow_gm_list'].append(float(np.mean(logger[str(order)][str(dim)][str(bond_dim)]['sym_pow_gm_list'])))

                logger[str(order)][str(dim)]['bond_dim_list'].append(bond_dim)
                if bond_dim_idx % write_freq == 0:
                    dump_object = json.dumps(logger)
                    with open(write_path, "w") as outfile:
                        outfile.write(dump_object)

                logger[str(order)][str(dim)][str(bond_dim)]['end_time'] = time.strftime('%X %x %Z')

            logger[str(order)]['dim_list'].append(dim)
            dump_object = json.dumps(logger)
            with open(write_path, "w") as outfile:
                outfile.write(dump_object)
    
        logger['order_list'].append(order)
        dump_object = json.dumps(logger)
        with open(write_path, "w") as outfile:
            outfile.write(dump_object)


    logger['exp_end_time'] = time.strftime('%X %x %Z')


def DickeExp(   num_particles, 
                alloc=3.2, write_freq=1,   write_path=None,
                device=None, config=def_config,   normalize=False):
    '''
    Dicke State experiment
    dim_list: values of d (>> values of n)
    order_list: values of n
    num_samples: number of samples for each (order, dim)
    batch_size: size of each batch < number of samples
    cmplx: bool for complex or real
    normalize: bool for normalizing tensors pre-optimzation
    '''

    utils.set_seed(config.seed)

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    if write_path is None:
        write_path = f"Logs/Dicke_{str(num_particles)}.json"
    
    logger = {}
    logger['exp_start_time'] = time.strftime('%X %x %Z')
    logger['alloc'] = alloc
    logger['partition_tuple_list'] = []
    logger['ana_gm_list'] = []
    logger['als_gm_list'] = []
    logger['ngd_gm_list'] = []
    logger['ngd_loss_list'] = []

    partition_tuples = []
    for part in range(1, int(num_particles//2)+1):
        print("="*75)
        partition_tuple = (part, num_particles-part)
        print(f"Tuple {part} of {int(num_particles//2)}: {partition_tuple}")

        pre_factor = math.factorial(num_particles)
        iter_factor = 1
        for part in partition_tuple:
            iter_factor *= ((part/num_particles)**part) * (1/math.factorial(part))
        
        logger['ana_gm_list'].append(pre_factor*iter_factor)

        non_state_batch = comp.make_Dicke(num_particles, partition_tuple)
        non_state_batch.data_[non_state_batch.data_==0] = 1e-6
        non_state_batch.data_ /=  torch.norm(non_state_batch.data_)
        assert torch.isclose(torch.norm(non_state_batch.data_), torch.tensor(1.0), rtol=config.rtol).item() == True

        # if cmplx == False:
        print("Starting ALS")
        non_als_cores = tn.Tensor(non_state_batch.data_, batch=True, ranks_cp=1).cores
        non_als_cores = [x.permute([0] + [x for x in range(1,len(x.shape))[::-1]]) for x in non_als_cores]
        non_als_approx = CandyComp(non_als_cores)
        for _ in range(1):
            non_als_approx[_] /= torch.norm(non_als_approx[_])
            assert torch.isclose(torch.norm(non_als_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
        logger['als_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', non_als_approx.view(1, -1), non_state_batch.data_.view(1, -1).conj()).view(-1,)).tolist())

        non_ngd_cores, non_ngd_loss = DeComp(non_state_batch, batched=True, config=config)
        non_ngd_approx = CandyComp(non_ngd_cores)
        for _ in range(1):
            non_ngd_approx[_] /= torch.norm(non_ngd_approx[_])
            assert torch.isclose(torch.norm(non_ngd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True

        logger['ngd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', non_ngd_approx.view(1, -1), non_state_batch.data_.view(1, -1).conj()).view(-1,)).tolist())
        logger['ngd_loss_list'].append(non_ngd_loss)
        
        del non_state_batch
        # if cmplx == False:
        del non_als_cores
        del non_ngd_cores
        del non_ngd_loss
        del non_ngd_approx
        torch.cuda.empty_cache()

        # if cmplx == False:

        logger['partition_tuple_list'].append(partition_tuple)
        
        dump_object = json.dumps(logger)
        with open(write_path, "w") as outfile:
            outfile.write(dump_object)


    logger['exp_end_time'] = time.strftime('%X %x %Z')


def AntisymExp(  dim_list, 
                alloc=3.2, write_freq=1,   write_path=None,
                device=None, config=def_config,   normalize=False):
    '''
    Dicke State experiment
    dim_list: values of d (>> values of n)
    order_list: values of n
    num_samples: number of samples for each (order, dim)
    batch_size: size of each batch < number of samples
    cmplx: bool for complex or real
    normalize: bool for normalizing tensors pre-optimzation
    '''

    utils.set_seed(config.seed)

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    if write_path is None:
        write_path = f"Logs/Antisym_{str(dim_list)}.json"
    
    logger = {}
    logger['exp_start_time'] = time.strftime('%X %x %Z')
    logger['alloc'] = alloc
    logger['dim_list'] = []
    logger['ana_gm_list'] = []
    logger['als_gm_list'] = []
    logger['ngd_gm_list'] = []
    logger['ngd_loss_list'] = []

    partition_tuples = []
    for d_idx, dim in enumerate(dim_list):
        print("="*75)
        print(f"Dim {d_idx+1} of {len(dim_list)}: {dim}")

        order = dim
        non_state_batch = comp.make_Antisym(dim)
        non_state_batch.data_[non_state_batch.data_==0] = 1e-9*torch.randn((1,1))

        logger['ana_gm_list'].append(1/math.factorial(order))

        assert torch.isclose(torch.norm(non_state_batch.data_), torch.tensor(1.0), rtol=config.rtol).item() == True

        print("Starting ALS")
        non_als_cores = tn.Tensor(non_state_batch.data_, batch=True, ranks_cp=1).cores
        non_als_cores = [x.permute([0] + [x for x in range(1,len(x.shape))[::-1]]) for x in non_als_cores]
        non_als_approx = CandyComp(non_als_cores)
        for _ in range(1):
            non_als_approx[_] /= torch.norm(non_als_approx[_])
            assert torch.isclose(torch.norm(non_als_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True
        logger['als_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', non_als_approx.view(1, -1), non_state_batch.data_.view(1, -1).conj()).view(-1,)).tolist())

        non_ngd_cores, non_ngd_loss = DeComp(non_state_batch, batched=True, config=config)
        non_ngd_approx = CandyComp(non_ngd_cores)
        for _ in range(1):
            non_ngd_approx[_] /= torch.norm(non_ngd_approx[_])
            assert torch.isclose(torch.norm(non_ngd_approx[_]), torch.tensor(1.0), rtol=config.rtol).item() == True

        logger['ngd_gm_list'].extend(torch.abs(torch.einsum('bi, bi -> b', non_ngd_approx.view(1, -1), non_state_batch.data_.view(1, -1).conj()).view(-1,)).tolist())
        logger['ngd_loss_list'].append(non_ngd_loss)
        
        del non_state_batch
        del non_als_cores
        del non_ngd_cores
        del non_ngd_loss
        del non_ngd_approx
        torch.cuda.empty_cache()
        
        logger['dim_list'].append(dim)
        
        dump_object = json.dumps(logger)
        with open(write_path, "w") as outfile:
            outfile.write(dump_object)

    
    logger['exp_end_time'] = time.strftime('%X %x %Z')


if __name__ == '__main__':
    NormedGaussExp(range(200, 0, -15), [3, 2], 20, config=def_config, cmplx=False)