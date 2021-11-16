"""Calculate metrics for CPD methods."""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5
) -> Tuple[int, int, int, int, List[int], List[int]]:
    """Evaluate metric for CPD on one sequence.
    We assume, that there is no more than one change index in data (so, either 0 or 1 change)

    :param y_true: true labels
    :param y_pred: predicted change probabilities
    :param threshold: if probability above threshold, the change detected
    :return: tuple of
        - number of true positives;
        - number of true negatives;
        - number of false positives;
        - number of false negatives;
        - list of detection delays;
        - list of false positive delays.
    """
    delay = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    true_positive = 0
    false_positive_delay = 0
    seq_len = y_true.shape[0]

    # find index with real change (if there is a change)
    index_real = torch.where(y_true != y_true[0])[0]

    # find index with predicted change
    index_detected = torch.where((y_pred > threshold).long() != y_true[0])[0]

    # if there is a change in sequence
    # we calculate TP, FP, where
    # TP - if predict change AFTER moment of real change
    # FP - if predict change BEFORE moment of real change
    if len(index_real) > 0:
        real_change_index = index_real[0]

        # if detect something
        if len(index_detected) > 0:
            # find predictions AFTER a real moment of change
            index_sub_detected = torch.where(index_detected >= real_change_index)[0]

            # if all predictions lie AFTER a real moment, increase TP and calculate delays
            if len(index_sub_detected) == len(index_detected):
                false_positive_delay = real_change_index.item()
                detected_change_index = index_detected[0]
                delay = (detected_change_index - real_change_index).item()
                true_positive += 1
            # else, calculate FP
            else:
                false_positive_delay = index_detected[0].item()
                delay = 0
                false_positive += 1
        # if there are no predicted changes, increase FN and set delays to maximum
        else:
            false_positive_delay = seq_len
            delay = (seq_len - real_change_index).item()
            false_negative += 1

    # for cases without real changes, calculate TN and FP
    # TN - if normal and predict no change
    # FP - if predict change in normal sequence

    else:
        # if we detect something, increase FP and calculate FP delay
        if len(index_detected) > 0:
            false_positive_delay = index_detected[0].item()
            false_positive += 1
        # if there is no predicted changes, increase TN and set FP delay to maximum
        else:
            false_positive_delay = seq_len
            true_negative += 1

    return (
        true_positive,
        true_negative,
        false_negative,
        false_positive,
        delay,
        false_positive_delay,
    )


def evaluate_metrics_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
    device: str = "cuda",
    verbose: bool = True,
    baseline: bool = True 
) -> Tuple[int, int, int, int, float, float]:
    """Calculate metrics for CPD.

    :param model: trained CPD model
    :param test_loader: dataloader with test data
    :param threshold: if probability above threshold, the change detected
    :param device: use cuda or cpu
    :param verbose: if True, print metrics
    :return: tuple of
        - number of true positives;
        - number of true negatives;
        - number of false positives;
        - number of false negatives;
        - mean of detection delays list
        - mean of false alarms delays list
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
    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        
        if baseline:
            test_out = [model(i) for i in test_inputs]
            test_out = torch.stack(test_out)
            print(test_out.shape)
        
        else:
            test_out = model(test_inputs)
            print(test_out.shape)
            
        for label, output in zip(test_labels.squeeze(), test_out.squeeze()):
            (
                tp_cur,
                tn_cur,
                fn_cur,
                fp_cur,
                delay_curr,
                fp_delay_curr,
            ) = evaluate_metrics(label, output, threshold)

            tp_number += tp_cur
            fp_number += fp_cur
            tn_number += tn_cur
            fn_number += fn_cur

            delay.append(delay_curr)
            fp_delay.append(fp_delay_curr)

    if verbose:
        print(
            "TP: {}, FP: {}, TN: {}, FN: {}, DELAY:{}, FP_DELAY:{}".format(
                tp_number,
                fp_number,
                tn_number,
                fn_number,
                np.mean(delay),
                np.mean(fp_delay),
            )
        )

    return tp_number, tn_number, fp_number, fn_number, np.mean(delay), np.mean(fp_delay)


def get_pareto_metrics_for_threshold(
    model: nn.Module,
    test_loader: DataLoader,
    threshold_list: List[float],
    device: str = "cuda",
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
        ) = evaluate_metrics_on_set(model, test_loader, threshold, device)

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
