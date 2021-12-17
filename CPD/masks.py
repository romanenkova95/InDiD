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
        
def get_mask( exp, d = None, SEQ_LEN = 64):

    if exp == 'diag':
        n_true = d*2-1
        return torch.as_tensor(scipy.sparse.diags([1]*n_true, [int((n_true - 1) / 2) - i for i in range(n_true)], 
                                                 shape=(SEQ_LEN, SEQ_LEN), dtype=bool).toarray())
    if exp == 'triang-1diag':
        return get_hessenberg_mask(SEQ_LEN, d, 0)
    if exp == 'triang-3diag':
        return get_hessenberg_mask1(SEQ_LEN, d, 0)
    if exp == 'triang':
        return torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), dtype=torch.bool)) 