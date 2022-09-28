import math
import numpy as np
import torch
import utils

from comp import CandyComp
from config import def_config
from tqdm import tqdm
from wrapper import Tensaur, CompSetList

'''
============================================
Enough Making, Now Breaking
============================================
'''

def DeComp(T, rank=1, batched=None, device=None, config=def_config, decomp_only_init=False, show_tq=False):
    #region

    print("Starting DeComp")    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if not isinstance(T, Tensaur):
        if batched == None:
            raise ValueError(f"Expected either batched information or Tensaur object.")
        else:
            T = Tensaur(T, batched)
    else:
        if batched != T.batched and (batched != None) and (T.batched != False):
            raise ValueError(f"Expected batched and T.batched to match but got {batched} and {T.batched}")
    
    # Initialization
    if config.initialization is None:
        initialization = torch.normal
        mean = 0
    else:
        print("Support for custom intializations will be added soon.")
        return

    # Extracting data and shapes
    T_data = T.data_.to(device)
    shape_tuple = T_data.shape
    core_list = []

    if torch.is_complex(T_data):
        dtype=torch.cfloat
    else:
        dtype=torch.float

    if batched:
        batch_size = shape_tuple[0]
        core_list = [initialization(mean, math.sqrt(1/dim), (batch_size, rank, dim), device=device, requires_grad=True, dtype=dtype) for dim in shape_tuple[1:]]
    else:
        core_list = [initialization(mean, math.sqrt(1/dim), (rank, dim), device=device, requires_grad=True, dtype=dtype) for dim in shape_tuple]

    loss = float("inf")
    loss_list = []    
    kick_in_idx = int(config.initial_patience*config.max_epoch/100)
    patience_idx = int(config.patience_perc*config.max_epoch/100)
    
    criterion = config.criterion
    if config.optimizier == 'SGD':
        optimizier = torch.optim.SGD(core_list, config.lr)
    if config.optimizier == 'ADAM':
        optimizier = torch.optim.Adam(core_list, config.lr)

    if show_tq:
        epoch_tq = tqdm(range(int(config.max_epoch)), leave=config.leave)
    else:
        epoch_tq = range(int(config.max_epoch))
    for epoch in epoch_tq:
        initial_mem = torch.cuda.memory_allocated(0)
        if show_tq:
            epoch_tq.set_description(f"Loss: {loss}")
        optimizier.zero_grad()

        approx = CandyComp(utils.normalize_compsetlist(core_list))
        
        loss = criterion(approx, T_data)
        loss_list.append(loss.item())

        # Probably keep "initial patience kick in" as a different variable
        if epoch > kick_in_idx:
            mean_loss = np.mean(np.array(loss_list[-patience_idx:]))
            max_deviation = np.max(np.abs(np.array(loss_list[-patience_idx:]) - mean_loss))
            if max_deviation < config.saturation_fac * mean_loss:
                print(f"\nLoss Stagnant at: {loss}\n")
                if config.ret_loss:
                    return CompSetList(utils.normalize_compsetlist([core.cpu() for core in core_list])), loss_list
                return CompSetList(utils.normalize_compsetlist([core.cpu() for core in core_list]))

        if loss < config.thresh_loss:
            print(f"\nConverged with loss: {loss}\n")
            if config.ret_loss:
                return CompSetList(utils.normalize_compsetlist([core.cpu() for core in core_list])), loss_list
            return CompSetList(utils.normalize_compsetlist([core.cpu() for core in core_list]))

        loss.backward()
        optimizier.step()
        if decomp_only_init and epoch==2:
            two_epoch_mem = torch.cuda.memory_allocated(0)
            print("Returning memory after one pass")
            return two_epoch_mem
    
        
    print(f"\nMaximum Epochs reached with loss: {loss}\n")
    if config.ret_loss:
        return CompSetList(utils.normalize_compsetlist([core.cpu() for core in core_list])), loss_list
    return CompSetList(utils.normalize_compsetlist([core.cpu() for core in core_list]))
    #endregion


def SymDeComp(T, rank=1, batched=None, device=None, config=def_config, show_tq=False):
    #region
    print("Starting SymDeComp")
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if not isinstance(T, Tensaur):
        if batched == None:
            raise ValueError(f"Expected either batched information or Tensaur object.")
        else:
            T = Tensaur(T, batched)
    else:
        if batched != T.batched:
            raise ValueError(f"Expected batched and T.batched to match but got {batched} and {T.batched}")
    
    # Extracting data and shapes
    T_data = T.data_.to(device)

    dim = T_data.shape[-1]
    batched = T.batched
    if batched:
        order = len(T_data.shape) - 1
        shape_tuple = (T.batch_size,) + (dim,)*order
        assert shape_tuple == tuple(T_data.shape)
        init_shape_tuple = (T.batch_size, 1, dim)
    else:
        order = len(T_data.shape)
        shape_tuple = (dim,)*order
        assert shape_tuple == tuple(T_data.shape)
        init_shape_tuple = (1, dim)

    # Initialization
    if config.initialization is None:
        initialization = torch.normal
        mean = 0
        stdv = math.sqrt(1/dim)
    else:
        print("Support for custom intializations will be added soon.")
        return

    if torch.is_complex(T_data):
        dtype = torch.cfloat
    else:
        dtype = torch.float
    
    a = initialization(mean, stdv, init_shape_tuple, dtype=dtype, requires_grad=True, device=device)
    
    loss = float("inf")
    loss_list = []    
    kick_in_idx = int(config.initial_patience*config.max_epoch/100)
    patience_idx = int(config.patience_perc*config.max_epoch/100)

    criterion = config.criterion
    if config.optimizier == 'SGD':
        optimizier = torch.optim.SGD([a], config.lr)
    if config.optimizier == 'ADAM':
        optimizier = torch.optim.Adam([a], config.lr)
    
    if show_tq:
        epoch_tq = tqdm(range(int(config.max_epoch)), leave=config.leave)
    else:
        epoch_tq = range(int(config.max_epoch))
    for epoch in epoch_tq:
        if show_tq:
            epoch_tq.set_description(f"Loss: {loss}")
        optimizier.zero_grad()

        approx = CandyComp(utils.normalize_compsetlist([a]*order))
        loss = criterion(approx, T_data)
        loss_list.append(loss.item())

        # Probably keep "initial patience kick in" as a different variable
        if epoch > kick_in_idx:
            mean_loss = np.mean(np.array(loss_list[-patience_idx:]))
            max_deviation = np.max(np.abs(np.array(loss_list[-patience_idx:]) - mean_loss))
            if max_deviation < config.saturation_fac * mean_loss:
                print(f"\nLoss Stagnant at: {loss}\n")
                if config.ret_loss:
                    return CompSetList(utils.normalize_compsetlist([a.cpu()]*order)), loss_list
                return CompSetList(utils.normalize_compsetlist([a.cpu()]*order))

        if loss < config.thresh_loss:
            print(f"\nConverged with loss: {loss}\n")
            if config.ret_loss:
                return CompSetList(utils.normalize_compsetlist([a.cpu()]*order)), loss_list
            return CompSetList(utils.normalize_compsetlist([a.cpu()]*order))

        loss.backward()
        optimizier.step()
        
    print(f"\nMaximum Epochs reached with loss: {loss}\n")
    if config.ret_loss:
        return CompSetList(utils.normalize_compsetlist([a.cpu()]*order)), loss_list
    return CompSetList(utils.normalize_compsetlist([a.cpu()]*order))
    #endregion


def SymPowDeComp(T, batched=None, device=None, config=def_config, show_tq=False):
    #region
    print("Starting SymPowDeComp")
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if not isinstance(T, Tensaur):
        if batched == None:
            raise ValueError(f"Expected either batched information or Tensaur object.")
        else:
            T = Tensaur(T, batched)
    else:
        if batched != T.batched:
            raise ValueError(f"Expected batched and T.batched to match but got {batched} and {T.batched}")
    
    # Extracting data and shapes
    T_data = T.data_.to(device)

    dim = T_data.shape[-1]
    batched = T.batched
    if batched:
        order = len(T_data.shape) - 1
        shape_tuple = (T.batch_size,) + (dim,)*order
        assert shape_tuple == tuple(T_data.shape)
        init_shape_tuple = (T.batch_size, 1, dim)
    else:
        order = len(T_data.shape)
        shape_tuple = (dim,)*order
        assert shape_tuple == tuple(T_data.shape)
        init_shape_tuple = (1, dim)

    # Initialization
    if config.initialization is None:
        initialization = torch.normal
        mean = 0
        stdv = math.sqrt(1/dim)
    else:
        print("Support for custom intializations will be added soon.")
        return

    a = initialization(mean, stdv, init_shape_tuple, device=device)

    if torch.is_complex(T_data):
        a_comp = 1j*torch.normal(mean, stdv, (1, dim), device=device)
        a = (a + a_comp) / math.sqrt(2)


    loss = float("inf")
    loss_list = []    
    kick_in_idx = int(config.initial_patience*config.max_epoch/100)
    patience_idx = int(config.patience_perc*config.max_epoch/100)

    criterion = config.criterion

    # Matricized version of T
    if batched:
        view_T = T_data.view(T.batch_size, -1, dim).to(device)
    else:
        view_T = T_data.view(-1, dim).to(device)

    if show_tq:
        epoch_tq = tqdm(range(int(config.max_epoch)), leave=config.leave)
    else:
        epoch_tq = range(int(config.max_epoch))
    for epoch in epoch_tq:
        if show_tq:
            epoch_tq.set_description(f"Loss: {loss}")
        a = utils.normalize_compsetlist([a]).data_[0]
        if batched:
            a = torch.einsum('...ij, ...jk -> ...ik', CandyComp([a]*(order-1)).view(T.batch_size, 1, -1), view_T)
        else:
            a = torch.einsum('...ij, ...jk -> ...ik', CandyComp([a]*(order-1)).view(1, -1), view_T)
        
        approx = CandyComp([a]*order)
        loss = criterion(approx, T_data)
        loss_list.append(loss.item())

        if epoch > kick_in_idx:
            mean_loss = np.mean(np.array(loss_list[-patience_idx:]))
            max_deviation = np.max(np.abs(np.array(loss_list[-patience_idx:]) - mean_loss))
            if max_deviation < config.saturation_fac * mean_loss:
                print(f"\nLoss Stagnant at: {loss}\n")
                if config.ret_loss:
                    return CompSetList(utils.normalize_compsetlist([a.cpu()]*order)), loss_list
                return CompSetList(utils.normalize_compsetlist([a.cpu()]*order))
                

        if loss < config.thresh_loss:
            print(f"\nConverged with loss: {loss}\n")
            if config.ret_loss:
                return CompSetList(utils.normalize_compsetlist([a.cpu()]*order)), loss_list
            return CompSetList(utils.normalize_compsetlist([a.cpu()]*order))

    print(f"\nMaximum Epochs reached with loss: {loss}\n")
    if config.ret_loss:
        return CompSetList(utils.normalize_compsetlist([a.cpu()]*order)), loss_list
    return CompSetList(utils.normalize_compsetlist([a.cpu()]*order))
    #endregion    

if __name__ == '__main__':
    utils.set_seed(1234)
    a = torch.normal(0, 1,(1, 10)) + 1j*torch.normal(0, 1,(1, 10))
    a = CandyComp([a]*1)
    print("A shape: ", a.shape)
    b, _ = DeComp(a, batched=False)
    print("B data len: ", len(b.data_))
    print("B data shapes: ", b.data_[0].shape)
    print("B data00 norm: ", torch.norm(b.data_[0]))
    print(b.data_[0].dtype)
    print(b.data_[0])
    print(torch.norm(b.data_[0]))