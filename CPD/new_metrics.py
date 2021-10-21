from typing import List, Tuple

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


def find_change(mask):
    with_change, change_ind = np.where(mask == 1)    
    change_ind_dict = {k: [] for k in range(0, mask.shape[0])}
    
    for (k, v) in zip(with_change, change_ind):
        change_ind_dict[k].append(v)

    return change_ind_dict

def find_first_change(change_ind_dict):
    new_change_ind_dict = {}
    for k in change_ind_dict.keys():
        try:
            new_change_ind_dict[k] = change_ind_dict[k][0]
        except:
            new_change_ind_dict[k] = -1
    return new_change_ind_dict

def calculate_errors(real_change_ind, predicted_change_ind, seq_len):
    TN, TP, FN, FP = 0, 0, 0, 0
    FP_delay, delay = [], []
    
    assert(len(real_change_ind) == len(predicted_change_ind))
        
    for (real, pred) in zip(real_change_ind.values(), predicted_change_ind.values()):
        if (real == pred) and (real == -1):
            TN += 1
            FP_delay.append(seq_len)
            delay.append(0)
            
        if (real != pred) and (pred == -1):
            FN += 1
            FP_delay.append(seq_len)
            delay.append(seq_len - real)
            
        if (real <= pred) and (real != -1):
            TP += 1
            FP_delay.append(real)
            delay.append(pred - real)            
            
        if (real > pred) and (real != -1) and (pred != -1):
            FP += 1
            FP_delay.append(pred)
            delay.append(0) 
            
        if (pred != -1) and (real == -1):
            FP += 1
            FP_delay.append(pred)
            delay.append(0)
            
        
    assert((TN + TP + FN + FP) == len(real_change_ind))            
    
    return TN, FP, FN, TP, FP_delay, delay

def calculate_metrics(true_labels, predictions):

    mask_real = ~true_labels.eq(true_labels[:, 0][0])
    mask_predicted = ~predictions.eq(true_labels[:, 0][0])
    seq_len = true_labels.shape[1]

    real_change_ind = find_change(mask_real)
    predicted_change_ind = find_change(mask_predicted)

    real_change_ind = find_first_change(real_change_ind)
    predicted_change_ind = find_first_change(predicted_change_ind)


    seq_len = len(true_labels[0, :])
    TN, FP, FN, TP, FP_delay, delay = calculate_errors(real_change_ind, predicted_change_ind, seq_len)
    confusion_matrix = (TN, FP, FN, TP)
    return confusion_matrix, FP_delay, delay


def evaluate_metrics_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[int, int, int, int, float, float]:
    """Calculate metrics for CPD.
    """
    # initialize metrics
    fp_number = 0
    fn_number = 0
    tp_number = 0
    tn_number = 0
    delay = []
    fp_delay = []

    # calculate metrics on set
    model.eval()
    device = model.device.type
    
    FP_delays = []
    delays = []
    conf_matrixes = np.array((0, 0, 0, 0))
    
    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = test_inputs.float().to(device), test_labels.to(device)
        test_out = model(test_inputs)
        try:
            test_out = test_out.squeeze(2)
        except:
            test_out = test_out.squeeze(1)            
        predictions = test_out > threshold
        confusion_matrix, FP_delay, delay = calculate_metrics(test_labels.detach().cpu(), predictions.detach().cpu())

        conf_matrixes += np.array(confusion_matrix)
        FP_delays.extend(FP_delay)
        delays.extend(delay)
        
    TN, FP, FN, TP = conf_matrixes
        
    if verbose:
        print(
            "TP: {}, FP: {}, TN: {}, FN: {}, DELAY:{}, FP_DELAY:{}".format(
                TP, FP, TN, FN,
                np.mean(delays),
                np.mean(FP_delays)
            )
        )

    return TP, TN, FP, FN, np.mean(delays), np.mean(FP_delays)

def get_pareto_metrics_for_threshold(
    model: nn.Module,
    test_loader: DataLoader,
    threshold_list: List[float],
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Get FP, FN, delays and FP delays metrics for set of thresholds.

    :param model: trained CPD model
    :param test_loader: dataloader with test data
    :param threshold_list: list of thresholds
    :param device: use cuda or cpu
    :return: tuple of
        - list of false positives;
        - list of false negatives;
        - list of delays detection;
        - list of FP delays;
    """
    fp_number_list = []
    fn_number_list = []
    delay_list = []
    fp_delay_list = []
    for threshold in threshold_list:
        (
            _,
            _,
            fp_number,
            fn_number,
            mean_delay,
            mean_fp_delay,
        ) = evaluate_metrics_on_set(model, test_loader, threshold, verbose)

        fp_number_list.append(fp_number)
        fn_number_list.append(fn_number)
        delay_list.append(mean_delay)
        fp_delay_list.append(mean_fp_delay)

    return fp_number_list, fn_number_list, delay_list, fp_delay_list


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
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(Sprime, S):
    """Compute the covering of a segmentation S by a segmentation Sprime.
    This follows equation (8) in Arbaleaz, 2010.
    >>> cover_single([{1, 2, 3}, {4, 5}, {6}], [{1, 2, 3}, {4, 5, 6}])
    0.8333333333333334
    >>> cover_single([{1, 2, 3, 4}, {5, 6}], [{1, 2, 3, 4, 5, 6}])
    0.6666666666666666
    >>> cover_single([{1, 2}, {3, 4}, {5, 6}], [{1, 2, 3}, {4, 5, 6}])
    0.6666666666666666
    >>> cover_single([{1, 2, 3, 4, 5, 6}], [{1}, {2}, {3}, {4, 5, 6}])
    0.3333333333333333
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C

# code from TCPDBench END


def get_change_idx(pred, threshold=None):
    cp = []
    if threshold:
        cp_ids = np.where((pred > threshold) == True)[0]
    else:
        cp_ids = np.where((pred != pred[0]))[0]
    if np.size(cp_ids) == 0:
        return cp
    cp.append(cp_ids[0])
    return cp

def cover(model, test_dataloader, threshold):
    cs = []
    device = model.device.type
    model.to(device)
    
    for inputs, labels in test_dataloader:
        seq_len = labels[0].shape[0]
        inputs = inputs.to(device)

        pred = model(inputs)
        try: 
            pred = pred.squeeze(2)
        except:
            pred = pred.squeeze(1)
        pred = pred.detach().cpu().numpy()
        
        labels = labels.detach().cpu().numpy()

        for i in range(len(pred)):
            p = get_change_idx(pred[i], threshold)
            l = get_change_idx(labels[i])
            
            if (len(l) != 0):
                if (len(p) != 0):
                    pred_S = partition_from_cps(p, seq_len)
                    true_S = partition_from_cps(l, seq_len)
                    cs.append(cover_single(pred_S, true_S))
                else:
                    cs.append(float(l[0] / seq_len))
            else:
                if len(p) == 0:
                    cs.append(1)
                else:
                    cs.append(float(p[0] / seq_len))
    cs = np.mean(cs)
    return cs

def F1_score(model, test_dataloader, threshold):
    TP, TN, FP, FN, _, _ = evaluate_metrics_on_set(model, test_dataloader, threshold, verbose=False)
    f1_score = TP / (TP + 0.5 * (FP + FN))
    return f1_score

from ruptures.metrics import precision_recall


def F1_score_ruptures(model, test_dataloader, threshold, margin=10):
    device = model.device.type
    model.to(device)

    precisions = []
    recalls = []
    
    for inputs, labels in test_dataloader:
        seq_len = labels[0].shape[0]
        inputs = inputs.to(device)

        pred = model(inputs)
        try: 
            pred = pred.squeeze(2)
        except:
            pred = pred.squeeze(1)
        pred = pred.detach().cpu().numpy()
        
        labels = labels.detach().cpu().numpy()
        
        for i in range(len(pred)):
            p = get_change_idx(pred[i], threshold)
            l = get_change_idx(labels[i])
            
            if (len(l) != 0):
                if (len(p) != 0):
                    precision, recall = precision_recall(p + [seq_len], l + [seq_len], margin)
                    precisions.append(precision)
                    recalls.append(recall)
                    
    macro_precisions = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1_score = 2.0 * macro_precisions * macro_recall / (macro_precisions + macro_recall)
    return macro_f1_score
            
