import torch

""""
============================================
Wrapper Classes
============================================
"""
#region


class Tensaur():
    def __init__(self, data, batched=False):
        '''
        data: iteratable of shape [b, d1, d2, ..., dn] or [d1, d2, ..., dn]
        '''
        if not isinstance(data, Tensaur):
            if not torch.is_tensor(data):
                data = torch.tensor(data)
            self.batch_size = None
            self.batched = False
            self.data_ = data
            if batched:
                self.batch_size = data.shape[0]
                self.batched = True
        else:
            self.__dict__.update(data.__dict__) 
    
    def to(self, device):
        self.data_ = self.data_.to(device)
        return self


class CompSetList():
    def __init__(self, data):
        '''
        data: list of iteratables of shape [b, r, d] or [r, d]
        attributes:
        data: data
        batched: bool
        batch_size: int if batched=True
        rank: rank of compsets (has to be same for all compsets)
        '''
        if not isinstance(data, CompSetList):
           
            data = [torch.tensor(x) if not torch.is_tensor(x) else x for x in data]

            # Shape compatability check
            for idx in range(1, len(data)):
                if len(data[idx].shape) > 3:
                    raise ValueError(f"Expected component-set shapes to be [batch_size, rank, d_i], [rank, d_i] or [d_i] but got {data[idx].shape} for element {idx}.")
                prev_idx = idx - 1
                if data[idx].shape[:-1] != data[prev_idx].shape[:-1]:
                    raise ValueError(f"Expected components to be consistent in batch and rank but got {data[prev_idx].shape} for element {prev_idx} and {data[idx].shape} for element {idx}.")
        
            self.data_ = data
            
            if len(data[0].shape) == 3:
                self.batched = True
                self.rank = data[0].shape[1]
                self.batch_size = data[0].shape[0]
            
            else:
                self.batched = False
                self.batch_size = None
                if len(data[0].shape) == 2:
                    self.rank = data[0].shape[0]
                elif len(data[0].shape) <= 1:
                    self.rank = 1
                    self.data_ = [x.view(1, -1) for x in data]

        else:
            self.__dict__.update(data.__dict__)        
#endregion
