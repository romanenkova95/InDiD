"""Module with functions for metrics calculation."""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import pickle
from pathlib import Path

import pytorch_lightning as pl

from utils import klcpd, tscp, cpd_models

#------------------------------------------------------------------------------------------------------------#
#                         Evaluate seq2seq, KL-CPD and TS-CP2 baseline models                                #
#------------------------------------------------------------------------------------------------------------#

def find_first_change(mask: np.array) -> np.array:
    """Find first change in batch of predictions.

    :param mask:
    :return: mask with -1 on first change
    """
    change_ind = torch.argmax(mask.int(), axis=1)
    no_change_ind = torch.sum(mask, axis=1)
    change_ind[torch.where(no_change_ind == 0)[0]] = -1
    return change_ind

def calculate_errors(
    real: torch.Tensor,
    pred: torch.Tensor,
    seq_len: int
) -> Tuple[int, int, int, int, List[float], List[float]]:
    """Calculate confusion matrix, detection delay and time to false alarms.

    :param real: real labels of change points
    :param pred: predicted labels (0 or 1) of change points
    :param seq_len: length of sequence
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
    """
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

def calculate_metrics(
    true_labels: torch.Tensor,
    predictions: torch.Tensor
) -> Tuple[int, int, int, int, np.array, np.array, int]:
    """Calculate confusion matrix, detection delay, time to false alarms, covering.

    :param true_labels: true labels (0 or 1) of change points
    :param predictions: predicted labels (0 or 1) of change points
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
        - covering
    """
    mask_real = ~true_labels.eq(true_labels[:, 0][0])
    mask_predicted = ~predictions.eq(true_labels[:, 0][0])
    seq_len = true_labels.shape[1]

    real_change_ind = find_first_change(mask_real)
    predicted_change_ind = find_first_change(mask_predicted)
    
    TN, FP, FN, TP, FP_delay, delay = calculate_errors(real_change_ind, predicted_change_ind, seq_len)
    cover = calculate_cover(real_change_ind, predicted_change_ind, seq_len)
        
    return TN, FP, FN, TP, FP_delay, delay, cover

def get_models_predictions(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    model_type: str = 'seq2seq',
    subseq_len: int = None,
    device: str = 'cuda',
    scale: int = None
) -> List[torch.Tensor]:
    """Get model's prediction.

    :param inputs: input data
    :param labels: true labels
    :param model: CPD model
    :param model_type: default "seq2seq" for BCE model, "klcpd" for KLCPD model
    :param device: device name
    :param scales: scale parameter for KL-CPD predictions
    :return: model's predictions
    """
    inputs = inputs.to(device)
    true_labels = labels.to(device)

    if model_type in ['simple', 'weak_labels']:
        outs = []
        true_labels = []
        for batch_n in range(inputs.shape[0]):
            inp = inputs[batch_n].to(device)
            lab = labels[batch_n].to(device)
            
            if model_type == 'simple':
                #TODO FIX
                #out = [model(inp[i].flatten().unsqueeze(0).float()).squeeze() for i in range(0, len(inp))]
                out = [model(inp[:, i].unsqueeze(0).float()).squeeze() for i in range(0, len(inp))]
                
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
    elif model_type == 'tscp':
        outs = tscp.get_tscp_output_scaled(model, inputs, model.window_1, model.window_2, scale=scale)
    elif model_type == 'kl_cpd':
        outs = klcpd.get_klcpd_output_scaled(model, inputs, model.window_1, model.window_2, scale=scale)
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
    device: str = 'cuda',
    scale: int = None
) -> Tuple[int, int, int, int, float, float]:
    """Calculate metrics for CPD.

    :param model: trained CPD model for evaluation
    :param test_loader: dataloader with test data
    :param threshold: alarm threshold (if change prob > threshold, report about a CP)
    :param verbose: if True, print the results
    :param model_type: type of the model ('seq2seq', 'kl_cpd', 'tscp', baselines)
    :param subseq_len: length of a subsequence (for 'weak_labels' baseline)
    :param device: 'cuda' or 'cpu'
    :param scale: scale factor (for KL-CPD and TSCP models)
    :return: tuple of
        - TN, FP, FN, TP
        - mean time to a false alarm
        - mean detection delay
        - mean covering
    """
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
                                                           device=device,
                                                           scale=scale)

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

def overlap(A: set, B: set):
    """Return the overlap (i.e. Jaccard index) of two sets.

    :param A: set #1
    :param B: set #2
    return Jaccard index of the 2 sets
    """
    return len(A.intersection(B)) / len(A.union(B))

def partition_from_cps(locations: List[int], n_obs: int) -> List[set]:
    """ Return a list of sets that give a partition of the set [0, T-1], as 
    defined by the change point locations.
    
    :param locations: idxs of the change points
    :param n_obs: length of the sequence
    :return partition of the sequence (list of sets with idxs)
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

def cover_single(
    true_partitions: List[set],
    pred_partitions: List[set]
) -> float:
    """Compute the covering of a true segmentation by a predicted segmentation.

    :param true_partitions: partition made by true CPs
    :param true_partitions: partition made by predicted CPs 
    """
    seq_len = sum(map(len, pred_partitions))
    assert seq_len == sum(map(len, true_partitions))
        
    cover = 0
    for t_part in true_partitions:
        cover += len(t_part) * max(overlap(t_part, p_part) for p_part in pred_partitions)
    cover /= seq_len
    return cover


def calculate_cover(
    real_change_ind: List[int],
    predicted_change_ind: List[int],
    seq_len: int
) -> List[float]:
    """Calculate covering for a given sequence.

    :param real_change_ind: indexes of true CPs
    :param predicted_change_ind: indexes of predicted CPs
    :param seq_len: length of the sequence
    :return cover
    """
    covers = []
    
    for real, pred in zip(real_change_ind, predicted_change_ind):
        true_partition = partition_from_cps([real.item()], seq_len)                
        pred_partition = partition_from_cps([pred.item()], seq_len)
        covers.append(cover_single(true_partition, pred_partition))
    return covers

def F1_score(confusion_matrix: Tuple[int, int, int, int]) -> float:
    """Calculate F1-score.

    :param confusion_matrix: tuple with elements of the confusion matrix
    :return: f1_score
    """
    TN, FP, FN, TP = confusion_matrix
    f1_score = 2.0 * TP / (2 * TP + FN + FP)
    return f1_score


def evaluation_pipeline(
    model: pl.LightningModule,
    test_dataloader: DataLoader,
    threshold_list: List[float],
    device: str = 'cuda', 
    verbose: bool = False, 
    model_type: str = 'seq2seq',
    subseq_len: int = None,
    scale: int = None
) -> Tuple[Tuple[float], dict, dict]:
    """Evaluate trained CPD model.

    :param model: trained CPD model to be evaluated
    :param test_dataloader: test data for evaluation
    :param threshold_list: listh of alarm thresholds
    :param device: 'cuda' or 'cpu'
    :param verbose: if True, print the results
    :param model_type: type of the model ('seq2seq', 'kl_cpd', 'tscp', baselines)
    :param subseq_len: subsequence length (for 'weak_labels' baseline)
    :param scale: scale factor (for KL-CPD and TSCP models)
    :return: tuple of
        - threshold th_1 corresponding to the maximum F1-score
        - mean time to a False Alarm corresponding to th_1
        - mean Detection Delay corresponding to th_1
        - Area under the Detection Curve
        - number of TN, FP, FN, TP corresponding to th_1
        - value of Covering corresponding to th_1
        - threshold th_2 corresponding to the maximum Covering metric
        - maximum value of Covering
    """
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
        ) = evaluate_metrics_on_set(
            model=model,
            test_loader=test_dataloader,
            threshold=threshold,
            verbose=verbose,
            model_type=model_type,
            subseq_len=subseq_len,
            device=device,
            scale=scale
            )

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

        
    return (best_th_f1, best_time_to_FA, best_delay, auc, best_conf_matrix, best_f1, best_cover, 
                                            best_th_cover, max_cover), delay_dict, fp_delay_dict


#------------------------------------------------------------------------------------------------------------#
#                                      Evaluate classic baselines                                            #
#------------------------------------------------------------------------------------------------------------#

def get_classic_baseline_predictions(
    dataloader: DataLoader,
    baseline_model: cpd_models.ClassicBaseline,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Get predictions of a classic baseline model.

    :param dataloader: validation dataloader
    :param baseline_model: core model of a classic baseline (from ruptures package)
    :return: tuple of
        - predicted labels
        - true pabels
    """
    all_predictions = []
    all_labels = []
    for inputs, labels in dataloader:
        all_labels.append(labels)
        baseline_pred = baseline_model(inputs)
        all_predictions.append(baseline_pred)

    all_labels = torch.from_numpy(np.vstack(all_labels))
    all_predictions = torch.from_numpy(np.vstack(all_predictions))
    return all_predictions, all_labels


def classic_baseline_metrics(
    all_labels: torch.Tensor,
    all_preds: torch.Tensor, 
    threshold: float=0.5
) -> Tuple[float, float, float, None, Tuple[int], float, float, float, float]:
    """ Calculate metrics for a classic baseline model.

    :param all_labels: tensor of true labels
    :param all_preds: tensor of predictions
    :param threshold: alarm threshold (=0.5 for classic models)
    :return: turple of metrics
        - best threshold for F1-score (always 0.5)
        - mean Time to a False Alarm
        - mean Detection Delay
        - None (no AUC metric for classic baselines)
        - best confusion matrix (number of TN, FP, FN and TP predictions)
        - F1-score
        - covering metric
        - best thresold for covering metric (always 0.5)
        - covering metric
    Note that we return some unnecessary values for consistency with our general evaluation pipeline.
    """
    FP_delays = []
    delays = []
    covers = []
    TN, FP, FN, TP = (0, 0, 0, 0)
    TN, FP, FN, TP, FP_delay, delay, cover = calculate_metrics(all_labels, all_preds > threshold)     
    f1 = F1_score((TN, FP, FN, TP))
    FP_delay = torch.mean(FP_delay.float()).item()
    delay = torch.mean(delay.float()).item()
    cover = np.mean(cover)
    return 0.5, FP_delay, delay, None, (TN, FP, FN, TP), f1, cover, 0.5, cover

def calculate_baseline_metrics(
    model: cpd_models.ClassicBaseline,
    val_dataloader: DataLoader,
    verbose: bool=False
) -> Tuple[float, float, float, None, Tuple[int], float, float, float, float]:
    """ Calculate metrics for a classic baseline model.

    :param model: core model of a classic baseline (from ruptures package)
    :param val_dataloader: validation dataloader
    :param verbose: if true, print the metrics to the console
    :return: tuple of metrics (see 'classic_baseline_metrics' function)
    """
    pred, labels = get_classic_baseline_predictions(val_dataloader, model)
    metrics = classic_baseline_metrics(labels, pred)

    _, mean_FP_delay, mean_delay, _, (TN, FP, FN, TP), f1, cover, _, _ = metrics

    if verbose:
        print(
            f'TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}, DELAY: {mean_delay}, FP_DELAY:{mean_FP_delay}, F1:{f1}, COVER: {cover}'
        )
    return metrics

#------------------------------------------------------------------------------------------------------------#
#                                              Save results                                                  #
#------------------------------------------------------------------------------------------------------------#
def write_metrics_to_file(
    filename: str,
    metrics: tuple,
    seed: int,
    timestamp: str
) -> None:
    """Write metrics to a .txt file.

    :param filename: path to the .txt file
    :param metrics: tuple of metrics (output of the 'evaluation_pipeline' function)
    :param seed: initialization seed for the model under evaluation
    :param timestamp: timestamp indicating which model was evaluated
    """
    best_th_f1, best_time_to_FA, best_delay, auc, best_conf_matrix, best_f1, best_cover, best_th_cover, max_cover = metrics
    
    with open(filename, 'a') as f:
        f.writelines('SEED: {}\n'.format(seed))
        f.writelines('Timestamp: {}\n'.format(timestamp))
        f.writelines('AUC: {}\n'.format(auc))
        f.writelines('Time to FA {}, delay detection {} for best-F1 threshold: {}\n'. format(round(best_time_to_FA, 4), 
                                                                                       round(best_delay, 4), 
                                                                                       round(best_th_f1, 4)))
        f.writelines('TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}\n'. format(best_conf_matrix[0],
                                                                               best_conf_matrix[1],
                                                                               best_conf_matrix[2],
                                                                               best_conf_matrix[3],
                                                                               round(best_th_f1, 4)))
        f.writelines('Max F1 {}: for best-F1 threshold {}\n'.format(round(best_f1, 4), round(best_th_f1, 4)))
        f.writelines('COVER {}: for best-F1 threshold {}\n'.format(round(best_cover, 4), round(best_th_f1, 4)))

        f.writelines('Max COVER {}: for threshold {}\n'.format(max_cover, best_th_cover))
        f.writelines('----------------------------------------------------------------------\n')

def dump_results(metrics_local: tuple, pickle_name: str) -> None:
    """Save result metrics as a .pickle file."""
    best_th_f1, best_time_to_FA, best_delay, auc, best_conf_matrix, best_f1, best_cover, best_th_cover, max_cover = metrics_local
    results = dict(
        best_th_f1=best_th_f1,
        best_time_to_FA=best_time_to_FA,
        best_delay=best_delay,
        auc=auc,
        best_conf_matrix=best_conf_matrix,
        best_f1=best_f1,
        best_cover=best_cover,
        best_th_cover=best_th_cover,
        max_cover=max_cover
        )

    with Path(pickle_name).open("wb") as f:
        pickle.dump(results, f)