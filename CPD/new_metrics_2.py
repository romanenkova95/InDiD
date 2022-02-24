from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc


def find_first_change(mask):    
    change_ind = torch.argmax(mask.int(), axis=1)
    no_change_ind = torch.sum(mask, axis=1)
    #change_ind = torch.where(change_ind==0, -1, change_ind)
    change_ind[torch.where(no_change_ind == 0)[0]] = -1
    return change_ind

def calculate_errors(real, pred, seq_len):
        
    FP_delay = torch.zeros_like(real, requires_grad=False)
    delay = torch.zeros_like(real, requires_grad=False)
    
    tn_mask = torch.logical_and(real==pred, real==-1)
    fn_mask = torch.logical_and(real!=pred, pred==-1)
    tp_mask = torch.logical_and(real<=pred, real!=-1)
    fp_mask = torch.logical_or(torch.logical_and(torch.logical_and(real>pred, real!=-1), pred!=-1),
                               torch.logical_and(pred!=-1, real==-1))

    TN = tn_mask.sum().item()
    FN = fn_mask.sum().item()
    TP = tp_mask.sum().item()
    FP = fp_mask.sum().item()

    FP_delay[tn_mask] = seq_len
    FP_delay[fn_mask] = seq_len
    FP_delay[tp_mask] = real[tp_mask] 
    FP_delay[fp_mask] = pred[fp_mask]

    delay[tn_mask] = 0
    delay[fn_mask] = seq_len - real[fn_mask]
    delay[tp_mask] = pred[tp_mask] - real[tp_mask]
    delay[fp_mask] = 0            
        
    assert((TN + TP + FN + FP) == len(real))      
    
    return TN, FP, FN, TP, FP_delay, delay

def calculate_metrics(true_labels, predictions):
    
    mask_real = ~true_labels.eq(true_labels[:, 0][0])
    mask_predicted = ~predictions.eq(true_labels[:, 0][0])
    seq_len = true_labels.shape[1]

    real_change_ind = find_first_change(mask_real)
    predicted_change_ind = find_first_change(mask_predicted)
    
    TN, FP, FN, TP, FP_delay, delay = calculate_errors(real_change_ind, predicted_change_ind, seq_len)
    cover = calculate_cover(real_change_ind, predicted_change_ind, seq_len)
        
    return TN, FP, FN, TP, FP_delay, delay, cover

##########################################
def get_models_predictions(inputs, labels, model, model_type='seq2seq', subseq_len=None, device='cuda'):

    inputs = inputs.to(device)
    true_labels = labels.to(device)

    if model_type in ['simple', 'weak_labels']:
        outs = []
        true_labels = []
        for batch_n in range(inputs.shape[0]):
            inp = inputs[batch_n].to(device)
            lab = labels[batch_n].to(device)
            
            if model_type == 'simple':
                out = [model(inp[i].flatten().unsqueeze(0).float()).squeeze() for i in range(0, len(inp))]
            elif (model_type == 'weak_labels') and (subseq_len is not None):
                out_end = [model(inp[i: i + subseq_len].flatten(1).unsqueeze(0).float()) for i in range(0, len(inp) - subseq_len)]
                out = [torch.zeros(len(lab) - len(out_end), 1, device=device)]                
                out.extend(out_end)
                out = torch.cat(out)
            true_labels += [lab]
            #TODO: fix
            try:
                outs.append(torch.stack(out))
            except:
                outs.append(out)
        outs = torch.stack(outs)
        true_labels = torch.stack(true_labels)                
    else:
        outs = model(inputs)
    return outs, true_labels

def evaluate_metrics_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
    verbose: bool = True,
    model_type: str = 'seq2seq',
    subseq_len: int = None, 
    device: str = 'cuda'
) -> Tuple[int, int, int, int, float, float]:
    """Calculate metrics for CPD.
    """
    # calculate metrics on set
    model.eval()
    model.to(device)    
    
    FP_delays = []
    delays = []
    covers = []
    TN, FP, FN, TP = (0, 0, 0, 0)
    
    with torch.no_grad():
            
        for test_inputs, test_labels in test_loader:
            test_out, test_labels = get_models_predictions(test_inputs, test_labels, 
                                                           model, 
                                                           model_type=model_type, 
                                                           subseq_len=subseq_len, 
                                                           device=device)

            try:
                test_out = test_out.squeeze(2)
            except:
                try:
                    test_out = test_out.squeeze(1)
                except:
                    test_out = test_out

            tn, fp, fn, tp, FP_delay, delay, cover = calculate_metrics(test_labels, test_out > threshold)     

            del test_labels
            del test_out
            gc.collect()
            if 'cuda' in device:
                torch.cuda.empty_cache() 

            TN += tn
            FP += fp
            FN += fn
            TP += tp
            
            FP_delays.append(FP_delay.detach().cpu())
            delays.append(delay.detach().cpu())
            covers.extend(cover)

                        
    mean_FP_delay = torch.cat(FP_delays).float().mean().item()
    mean_delay = torch.cat(delays).float().mean().item()
    mean_cover = np.mean(covers)
                   
    if verbose:
        print(
            "TN: {}, FP: {}, FN: {}, TP: {}, DELAY:{}, FP_DELAY:{}, COVER: {}".format(
                TN, FP, FN, TP ,
                mean_delay,
                mean_FP_delay,
                mean_cover
            )
        )
        
    del FP_delays
    del delays
    del covers
    
    gc.collect()
    if 'cuda' in device:
        torch.cuda.empty_cache()         

    return TN, FP, FN, TP, mean_delay, mean_FP_delay, mean_cover

def area_under_graph(delay_list: List[float], fp_delay_list: List[float]) -> float:
    
    """Calculate area under Delay - FP delay curve.

    :param delay_list: list of delays
    :param fp_delay_list: list of fp delays
    :return: area under curve
    """
    return np.trapz(delay_list, fp_delay_list)


#-------------------------------------------------------------------------------------------------------------------------------
# code from TCPDBench START

def overlap(A, B):
    """ Return the overlap (i.e. Jaccard index) of two sets
    >>> overlap({1, 2, 3}, set())
    0.0
    >>> overlap({1, 2, 3}, {2, 5})
    0.25
    >>> overlap(set(), {1, 2, 3})
    0.0
    >>> overlap({1, 2, 3}, {1, 2, 3})
    1.0
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    """ Return a list of sets that give a partition of the set [0, T-1], as 
    defined by the change point locations.
    >>> partition_from_cps([], 5)
    [{0, 1, 2, 3, 4}]
    >>> partition_from_cps([3, 5], 8)
    [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    >>> partition_from_cps([1,2,7], 8)
    [{0}, {1}, {2, 3, 4, 5, 6}, {7}]
    >>> partition_from_cps([0, 4], 6)
    [{0, 1, 2, 3}, {4, 5}]
    """
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(n_obs):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(true_partitions, pred_partitions):
    """Compute the covering of a true segmentation by a predicted segmentation.
    """
    seq_len = sum(map(len, pred_partitions))
    assert seq_len == sum(map(len, true_partitions))
        
    cover = 0
    for t_part in true_partitions:
        cover += len(t_part) * max(overlap(t_part, p_part) for p_part in pred_partitions)
    cover /= seq_len
    return cover


def calculate_cover(real_change_ind, predicted_change_ind, seq_len):
    covers = []
    
    for real, pred in zip(real_change_ind, predicted_change_ind):
        true_partition = partition_from_cps([real.item()], seq_len)                
        pred_partition = partition_from_cps([pred.item()], seq_len)
        covers.append(cover_single(true_partition, pred_partition))
    
    return covers
        
        
def F1_score(confusion_matrix):
    TN, FP, FN, TP = confusion_matrix
    f1_score = 2.0 * TP / (2 * TP + FN + FP)
    return f1_score


#########################################################################################
def evaluation_pipeline(model, test_dataloader, threshold_list, device='cuda', verbose=False, 
                        model_type='seq2seq', subseq_len=None):
    try:
        model.to(device)
        model.eval()
    except:
        print('Cannot move model to device')    
    
    cover_dict = {}
    f1_dict = {}
    f1_lib_dict = {}
    
    delay_dict = {}
    fp_delay_dict = {}
    confusion_matrix_dict = {}    

    for threshold in tqdm(threshold_list):
        (
            TN,
            FP,
            FN,
            TP,
            mean_delay,
            mean_fp_delay,
            cover
        ) = evaluate_metrics_on_set(model=model, test_loader=test_dataloader, threshold=threshold, 
                                    verbose=verbose, model_type=model_type, device=device)

        confusion_matrix_dict[threshold] = (TN, FP, FN, TP)
        delay_dict[threshold] = mean_delay
        fp_delay_dict[threshold] = mean_fp_delay        
        
        cover_dict[threshold] = cover
        f1_dict[threshold] = F1_score((TN, FP, FN, TP))
            
    auc = area_under_graph(list(delay_dict.values()), list(fp_delay_dict.values()))

    # Conf matrix and F1
    best_th_f1 = max(f1_dict, key=f1_dict.get)
 
    best_conf_matrix = (confusion_matrix_dict[best_th_f1][0], confusion_matrix_dict[best_th_f1][1], 
                        confusion_matrix_dict[best_th_f1][2], confusion_matrix_dict[best_th_f1][3])
    best_f1 = f1_dict[best_th_f1]
    
    # Cover
    best_cover = cover_dict[best_th_f1]
    
    best_th_cover = max(cover_dict, key=cover_dict.get)
    max_cover = cover_dict[best_th_cover]
    
    # Time to FA, detection delay
    best_time_to_FA = fp_delay_dict[best_th_f1]
    best_delay = delay_dict[best_th_f1]

    
    if verbose:
        print('AUC:', round(auc, 4))
        print('Time to FA {}, delay detection {} for best-F1 threshold: {}'. format(round(best_time_to_FA, 4), 
                                                                                       round(best_delay, 4), 
                                                                                       round(best_th_f1, 4)))
        print('TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}'. format(best_conf_matrix[0],
                                                                             best_conf_matrix[1],
                                                                             best_conf_matrix[2],
                                                                             best_conf_matrix[3],
                                                                             round(best_th_f1, 4)))
        print('Max F1 {}: for best-F1 threshold {}'.format(round(best_f1, 4), round(best_th_f1, 4)))
        print('COVER {}: for best-F1 threshold {}'.format(round(best_cover, 4), round(best_th_f1, 4)))

        print('Max COVER {}: for threshold {}'.format(round(cover_dict[max(cover_dict, key=cover_dict.get)], 4), 
                                                      round(max(cover_dict, key=cover_dict.get), 4)))

        
    return (best_th_f1, best_time_to_FA, best_delay, auc, best_conf_matrix, best_f1, best_cover, best_th_cover, max_cover), delay_dict, fp_delay_dict
