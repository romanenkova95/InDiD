from typing import List, Tuple

import numpy as np
import torch
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

##########################################
def get_models_predictions(inputs, labels, model, model_type='seq2seq', subseq_len=None, device='cuda'):
    inputs = inputs.to('cuda')
    model.to(device)
    
    if model_type in ['simple', 'weak_labels']:
        outs = []
        true_labels = []
        for batch_n in range(inputs.shape[0]):
            inp = inputs[batch_n]#.to(device)
            lab = labels[batch_n]#.to(device)
            
            if model_type == 'simple':
                out = [model(inp[i].flatten().unsqueeze(0).float()).squeeze() for i in range(0, len(inp))]
                true_labels += [lab]
            elif (model_type == 'weak_labels') and (subseq_len is not None):
                out = [model(inp[i: i + subseq_len].flatten(1).unsqueeze(0).float()).squeeze() for i in range(0, len(inp) - subseq_len)]
                true_labels += lab[(len(lab) - len(out)):].unsqueeze(0)        
            outs.append(torch.stack(out))                    
        outs = torch.stack(outs)
        true_labels = torch.stack(true_labels)                
    else:
        outs = model(inputs)
        true_labels = labels
    return outs, true_labels

def evaluate_metrics_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
    verbose: bool = True,
    model_type: str = 'seq2seq',
    subseq_len: int = None
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
    
    try:
        device = model.device.type
    except:
        try:
            device = model.device
        except:
            device = 'cpu'
    FP_delays = []
    delays = []
    conf_matrixes = np.array((0, 0, 0, 0))
    
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
    verbose: bool = True,
    baseline: bool = False
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
    tp_number_list = []
    tn_number_list = []    
    delay_list = []
    fp_delay_list = []
    for threshold in threshold_list:
        (
            tp_number,
            tn_number,
            fp_number,
            fn_number,
            mean_delay,
            mean_fp_delay,
        ) = evaluate_metrics_on_set(model, test_loader, threshold, verbose, baseline)

        tp_number_list.append(tp_number)
        tn_number_list.append(tn_number)                
        fp_number_list.append(fp_number)
        fn_number_list.append(fn_number)
        delay_list.append(mean_delay)
        fp_delay_list.append(mean_fp_delay)
        
    conf_matrix = (tp_number_list, tn_number_list, fp_number_list, fn_number_list)
    return conf_matrix, delay_list, fp_delay_list        


def area_under_graph(delay_list: List[float], fp_delay_list: List[float]) -> float:
    """Calculate area under Delay - FP delay curve.

    :param delay_list: list of delays
    :param fp_delay_list: list of fp delays
    :return: area under curve
    """
    return np.trapz(delay_list, fp_delay_list)


# (n_samples, n_dims)

import ruptures as rpt  # our package

def evaluate_baseline(dataloader, baseline_model, pen=None, n_pred=None):
    all_predictions = []
    all_labels = []
    for inputs, labels in dataloader:
        for i, seq in enumerate(inputs):
            signal = seq.flatten(1, 2).detach().numpy()
            label = labels[i]            
            algo = baseline_model.fit(signal)
            if pen:
                cp_pred = algo.predict(pen=pen)
            elif n_pred:
                cp_pred = algo.predict(n_pred)                
            cp_pred = cp_pred[0]
            baselines_pred = np.zeros(inputs.shape[1])
            baselines_pred[cp_pred:] = np.ones(inputs.shape[1] - cp_pred)        
            all_predictions.append(baselines_pred)
            all_labels.append(label)
    return all_predictions, all_labels


def baseline_metrics(all_labels, all_preds):
    fp_number = 0
    fn_number = 0
    tp_number = 0
    tn_number = 0
    delay = []
    fp_delay = []

    for label, output in zip(all_labels, all_preds):
        output = torch.from_numpy(output)
        (
            tp_cur,
            tn_cur,
            fn_cur,
            fp_cur,
            delay_curr,
            fp_delay_curr,
        ) = metrics.evaluate_metrics(label, output, 0.5)

        tp_number += tp_cur
        fp_number += fp_cur
        tn_number += tn_cur
        fn_number += fn_cur

        delay.append(delay_curr)
        fp_delay.append(fp_delay_curr)
        
        confusion_matrix = (tp_number, fp_number, tn_number, fn_number)
    return confusion_matrix, np.mean(delay), np.mean(fp_delay)


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


def cover(model, test_dataloader, threshold, model_type, subseq_len):
    cs = []
    try:
        device = model.device.type
    except:
        device = model.device
    model.to(device)
    
    for inputs, labels in test_dataloader:
        seq_len = labels[0].shape[0]
        inputs = inputs.to(device)

        test_out, labels = get_models_predictions(inputs, labels, 
                                                  model, 
                                                  model_type=model_type, 
                                                  subseq_len=subseq_len, 
                                                  device=device)
        try: 
            pred = test_out.squeeze(2)
        except:
            pred = test_out.squeeze(1)
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

def F1_score(model, test_dataloader, threshold, model_type, subseq_len):
    TP, TN, FP, FN, _, _ = evaluate_metrics_on_set(model, test_dataloader, threshold, verbose=False, model_type=model_type, subseq_len=subseq_len)
    f1_score = TP / (TP + 0.5 * (FP + FN))
    return f1_score

from ruptures.metrics import precision_recall


def F1_score_ruptures(model, test_dataloader, threshold, margin=10):
    try:
        device = model.device.type
    except:
        device = model.device
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
            
#########################################################################################
def evaluation_pipeline(model, test_dataloader, threshold_list, device='cuda', verbose=False, 
                        model_type='seq2seq', subseq_len=None):
    try:
        model.to(device)
        model.eval()
    except:
        print('Cannot move model to device')    
    
    fp_number_list = []
    fn_number_list = []
    tp_number_list = []
    tn_number_list = []    
    delay_list = []
    fp_delay_list = []
    
    
    cover_dict = {}
    f1_dict = {}
    f1_lib_dict = {}

    for th in threshold_list:
        (
            tp_number,
            tn_number,
            fp_number,
            fn_number,
            mean_delay,
            mean_fp_delay,
        ) = evaluate_metrics_on_set(model, test_dataloader, th, verbose=False, 
                                    model_type=model_type, subseq_len=subseq_len)
        
        tp_number_list.append(tp_number)
        tn_number_list.append(tn_number)                
        fp_number_list.append(fp_number)
        fn_number_list.append(fn_number)
        delay_list.append(mean_delay)
        fp_delay_list.append(mean_fp_delay)
        
        if (th <= 1) and (th > 0):
            cover_dict[th] = cover(model, test_dataloader, th, model_type=model_type, subseq_len=subseq_len)
            f1_dict[th] = F1_score(model, test_dataloader, th, model_type=model_type, subseq_len=subseq_len)
            #f1_lib_dict[th] = F1_score_ruptures(model, test_dataloader, th, margin=5)
            f1_lib_dict[th] = 0
            
    conf_matrix = (tp_number_list, tn_number_list, fp_number_list, fn_number_list)
    auc = area_under_graph(delay_list, fp_delay_list)

    # Cover
    best_th_cover = max(cover_dict, key=cover_dict.get)
    best_cover = cover_dict[best_th_cover]
    
    # Time to FA, detection delay
    ind = threshold_list.index(best_th_cover)
    best_time_to_FA = fp_delay_list[ind]
    best_delay = delay_list[ind]

    # Conf matrix and F1
    best_conf_matrix = (conf_matrix[0][ind], conf_matrix[1][ind], conf_matrix[2][ind], conf_matrix[3][ind])
    best_f1 = f1_dict[best_th_cover]
    
    # F1 from ruptures 
    best_f1_ruptures = f1_lib_dict[best_th_cover]
    
    if verbose:
        print('AUC:', round(auc, 4))
        print('Time to FA {}, delay detection {} for best-cover threshold: {}'. format(round(best_time_to_FA, 4), 
                                                                                       round(delay_list[ind], 4), 
                                                                                       round(best_th_cover, 4)))
        print('TP {}, TN {}, FP {}, FN {} for best-cover threshold: {}'. format(best_conf_matrix[0],
                                                                                best_conf_matrix[1],
                                                                                best_conf_matrix[2],
                                                                                best_conf_matrix[3],
                                                                                round(best_th_cover, 4)))
        print('Max COVER {}: for threshold {}'.format(round(best_cover, 4), 
                                                      round(best_th_cover, 4)))

        print('Max F1 {}: for threshold {}'.format(round(f1_dict[max(f1_dict, key=f1_dict.get)], 4), 
                                                   round(max(f1_dict, key=f1_dict.get), 4)))

        print('F1 {}: for best-cover threshold {}'.format(round(best_f1, 4), round(best_th_cover, 4)))
        print('Max F1_ruptures (M=5) {}: for threshold {}'.format(round(f1_lib_dict[max(f1_lib_dict, key=f1_lib_dict.get)], 4), 
                                                                  round(max(f1_lib_dict, key=f1_lib_dict.get), 4)))

        print('F1_ruptures {}: for best-cover threshold {}'.format(round(best_f1_ruptures, 4), round(best_th_cover, 4)))

    return (best_th_cover, best_time_to_FA, best_delay, auc, conf_matrix, best_f1, best_f1_ruptures, best_cover), delay_list, fp_delay_list
