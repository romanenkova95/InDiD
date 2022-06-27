"""Module with attention masks for network models with transformer"""

import torch
import torch.nn as nn
import scipy.sparse

def get_hessenberg_mask(seq_len, low_n, up_n):
    if low_n + 2 <= seq_len:
        position = [-seq_len + i for i in range(low_n + 1)]+[-1, 0, 1] +  [seq_len - i for i in range(up_n + 1)]
        mask = scipy.sparse.diags([1]*len(position), position, shape=(seq_len, seq_len), dtype = bool).toarray()
        return torch.as_tensor(mask)
    else:
        print('Error: low_n is too large, that is why offset array contains duplicate values. Try low_n <= seq_len - 2')
        
def get_hessenberg_mask1(seq_len, low_n, up_n):
    if low_n + 1 <= seq_len:
        position = [-seq_len + i for i in range(low_n + 1)]+[ 0] +  [seq_len - i for i in range(up_n + 1)]
        mask = scipy.sparse.diags([1]*len(position), position, shape=(seq_len, seq_len), dtype = bool).toarray()
        return torch.as_tensor(mask)
    else:
        print('Error: low_n is too large, that is why offset array contains duplicate values. Try low_n <= seq_len - 1')
        
def get_hessenberg_mask_wo_diag(seq_len, low_n, up_n):
    if low_n <= seq_len:
        position = [-seq_len + i for i in range(low_n + 1)]+[seq_len - i for i in range(up_n + 1)]
        mask = scipy.sparse.diags([1]*len(position), position, shape=(seq_len, seq_len), dtype = bool).toarray()
        return torch.as_tensor(mask)
    else:
        print('Error: low_n is too large, that is why offset array contains duplicate values. Try low_n <= seq_len ')

def get_lowtriang_and_lowdiag(seq_len, low_n, diag_n):
    if low_n + diag_n <= seq_len:
        position = [-seq_len + i for i in range(low_n + 1)]+ [ - i for i in range(diag_n)] 
        mask = scipy.sparse.diags([1]*len(position), position, shape=(seq_len, seq_len), dtype = bool).toarray()
        return torch.as_tensor(mask)
    else:
        print('Error: low_n + diag_n is too large, that is why offset array contains duplicate values. Try low_n + diag_n <= seq_len ')
def get_uptriang_and_updiag(seq_len, up_n, diag_n):
    if up_n + diag_n <= seq_len:
        position = [seq_len - i for i in range(up_n + 1)]+ [i for i in range(diag_n)] 
        mask = scipy.sparse.diags([1]*len(position), position, shape=(seq_len, seq_len), dtype = bool).toarray()
        return torch.as_tensor(mask)
    else:
        print('Error: up_n + diag_n is too large, that is why offset array contains duplicate values. Try up_n + diag_n <= seq_len ')
import torch
def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

def get_mask( exp, d = None, SEQ_LEN = 64, alpha=0.5):
    '''
    Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions. 
    If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
    If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged.
    If a FloatTensor is provided, it will be added to the attention weight.
    '''
    if exp == 'triang':
        return torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=bool), diagonal=1, )
    elif exp == 'diag':
        n_true = d*2-1
        return ~torch.as_tensor(scipy.sparse.diags([1]*n_true, [int((n_true - 1) / 2) - i for i in range(n_true)],  shape=(SEQ_LEN, SEQ_LEN), dtype=bool).toarray())
    elif exp == 'low-diag':
        return ~torch.as_tensor(scipy.sparse.diags([1]*(d), [ - i for i in range(d)],  shape=(SEQ_LEN, SEQ_LEN), dtype=bool).toarray())
    
    elif exp == 'triang-3diag':
        return ~get_hessenberg_mask(SEQ_LEN, d, 0)
    elif exp == 'triang-1diag':
        return ~get_hessenberg_mask1(SEQ_LEN, d, 0)
    elif exp == 'low-triang':
        return ~get_hessenberg_mask_wo_diag(SEQ_LEN, d, 0)
    
    elif exp == 'lowrandom':
        # alpha = d
        # alpha = 1 -- full triang mask, alpha = 0 -- full true mask
        a = (torch.triu(torch.ones(SEQ_LEN, SEQ_LEN)*2., diagonal=1, ) + torch.rand((SEQ_LEN, SEQ_LEN))) + torch.diag(torch.ones(SEQ_LEN))
        return (a < 1 - d) + (a > 2.) 
    elif exp == 'random':
        print('alpha =', d)
        # alpha = 0 -- diag, alpha = 1 -- all false
        a = torch.rand((SEQ_LEN, SEQ_LEN))
        a += torch.diag(torch.ones(SEQ_LEN))
        return a < 1. - d
    elif exp == 'trianglesides':
        a = torch.ones(SEQ_LEN, SEQ_LEN) > 0
        a[:, :1] = False
        a[-1:, :] = False
        a.fill_diagonal_(False)
        return a
    elif exp == 'lowdiag-lowtriang':
        return ~get_lowtriang_and_lowdiag(SEQ_LEN, d, d)
        
    elif exp == 'future-triang-1diag':
        return ~get_hessenberg_mask1(SEQ_LEN, 0, d)
    elif exp == 'up-diag':
        return ~torch.as_tensor(scipy.sparse.diags([1]*(d), [i for i in range(d)],  shape=(SEQ_LEN, SEQ_LEN), dtype=bool).toarray())
    elif exp == 'up-random':
        a = (torch.triu(torch.rand((SEQ_LEN, SEQ_LEN)), diagonal=1, )  + torch.diag(torch.ones(SEQ_LEN)*0.5))
        return (a == 0 ) + (a > alpha)
    elif exp == 'updiag-uptriang':
        return ~get_uptriang_and_updiag(SEQ_LEN, d, d)
    
    elif exp == 'lowtriang-updiag':
        return (torch.triu(torch.ones((SEQ_LEN, SEQ_LEN), dtype = bool), diagonal=d, ))
    elif exp == 'lowtriang-uprandom':
        a = (torch.triu(torch.rand((SEQ_LEN, SEQ_LEN)), diagonal=1, ) )
        return ~ ((a == 0 ) + (a > alpha))
    elif exp == 'blockdiag':
        if SEQ_LEN % d == 0:
            return ~(block_diag(torch.ones(SEQ_LEN//d, d, d)) > 0)
        else:
            print('SEQ_LEN mast be devided by d')
