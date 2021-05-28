import torch
import numpy as np
import pandas as pd
import os

columns_list = ["Model name", "Mean FP delay", "Mean delay", "Threshold", "TP", "TN", "FP", "FN", "Acc",
                "Precision", "Recall", "F1-score", "G-mean"]

# TP - если 1 после момента смены
# TN - если 0 до момента смены
# FP - если 1 до момента смены
# FN - если все 0 после момента смены

def evaluate_metrics(y_true, y_pred, seq_len, threshold=0.5):
    """
    Evaluate metrics for change point detection
    We assume, that there is no more than one change index in data (so, either 0 or 1 change)
    Inputs
    y_true : torch.Tensor
      true labels
    y_pred : torch.Tensor
      change probabiltiy
    threshold : float
      detection threshold

    Returns
    false_positive : int
      number of false positives
    false_negative : int
      number of false negatives
    delay : int
      detection delay
    accuracy : float
      y_pred accuracy given y_truei
    """

    delay = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    true_positive = 0
    false_positive_delay = 0
    
    index_real = torch.where(y_true != y_true[0])[0]  # indexes with real changes
    index_detected = torch.where((y_pred > threshold).long() != y_true[0])[0]  # indexes with predicted changes

    if len(index_real) > 0:
        real_change_index = index_real[0]

        if len(index_detected) > 0:
            index_sub_detected = torch.where(index_detected >= real_change_index)[0]

            if len(index_sub_detected) == len(index_detected):
                false_positive_delay = real_change_index.item()
                detected_change_index = index_detected[0]
                delay = (detected_change_index - real_change_index).item()
                true_positive += 1
            else:
                false_positive_delay = index_detected[0].item()
                delay = 0
                false_positive += 1

        else:
            false_positive_delay = seq_len
            delay = (seq_len - real_change_index).item()
            false_negative += 1

    else:
        if len(index_detected) > 0:
            false_positive_delay = index_detected[0].item()
            false_positive += 1
        else:
            false_positive_delay = seq_len
            true_negative += 1

    return true_positive, true_negative, false_negative, false_positive, delay, false_positive_delay


def evaluate_metrics_new(y_true, y_pred, seq_len, threshold=0.5):
    """
    Evaluate metrics for change point detection
    We assume, that there is no more than one change index in data (so, either 0 or 1 change)
    Inputs
    y_true : torch.Tensor
      true labels
    y_pred : torch.Tensor
      change probabiltiy
    threshold : float
      detection threshold

    Returns
    false_positive : int
      number of false positives
    false_negative : int
      number of false negatives
    delay : int
      detection delay
    accuracy : float
      y_pred accuracy given y_truei
    """

    delay = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    true_positive = 0
    false_positive_delay = 0

    index_real = torch.where(y_true != y_true[0])[0]  # indexes with real changes
    index_detected = torch.where((y_pred > threshold).long() != y_true[0])[0]  # indexes with predicted changes

    if len(index_real) > 0:
        real_change_index = index_real[0]

        if len(index_detected) > 0:
            index_sub_detected = torch.where(index_detected >= real_change_index)[0]

            if len(index_sub_detected) == len(index_detected):
                false_positive_delay = seq_len                
                detected_change_index = index_detected[0]
                delay = (detected_change_index - real_change_index).item()
                true_positive += 1
            else:
                false_positive_delay = index_detected[0].item()
                false_positive += 1

        else:
            false_positive_delay = seq_len                            
            delay = (seq_len - real_change_index).item()
            false_negative += 1

    else:
        if len(index_detected) > 0:
            false_positive_delay = index_detected[0].item()
            false_positive += 1
        else:
            false_positive_delay = seq_len                                        
            true_negative += 1

    return true_positive, true_negative, false_negative, false_positive, delay, false_positive_delay


def evaluate_metrics_on_set(model, test_loader, batch_size, seq_len, threshold=0.5, device='cuda', verbose=True):
    test_losses = []
    test_outputs = []
    overall_test_loss_list = []
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    delay = []
    fp_delay = []
    model.eval()
    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

        if test_inputs.shape[0] < batch_size:
            break

        test_out = model(test_inputs)
        test_outputs += [test_out.cpu().clone().detach()]

        for l, o in zip(test_labels.squeeze(), test_out.squeeze()):
            tp_cur, tn_cur, fn_cur, fp_cur, delay_curr, fp_delay_curr = evaluate_metrics(l, o, seq_len, threshold)
            tp += tp_cur
            fp += fp_cur
            tn += tn_cur
            fn += fn_cur

            delay.append(delay_curr)
            fp_delay.append(fp_delay_curr)

    overall_test_loss_list.append(np.mean(test_losses))

    if verbose:
        print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn, "DELAY:", np.mean(delay), "FP_DELAY", np.mean(fp_delay))
    return tp, tn, fp, fn, np.mean(delay), np.mean(fp_delay)


def evaluate_metrics_on_set_new(model, test_loader, batch_size, seq_len, threshold, device='cuda', verbose=True):

    test_losses = []
    test_outputs = []
    overall_test_loss_list = []
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    delay = []
    fp_delay = []
    model.eval()
    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

        if test_inputs.shape[0] < batch_size:
            break

        test_out = model(test_inputs)
        test_outputs += [test_out.cpu().clone().detach()]

        for l, o in zip(test_labels.squeeze(), test_out.squeeze()):
            tp_cur, tn_cur, fn_cur, fp_cur, delay_curr, fp_delay_curr = evaluate_metrics_new(l, o, seq_len, threshold)

            tp += tp_cur
            fp += fp_cur
            tn += tn_cur
            fn += fn_cur

            delay.append(delay_curr)
            fp_delay.append(fp_delay_curr)

    overall_test_loss_list.append(np.mean(test_losses))

    if verbose:
        print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn, "DELAY:", np.mean(delay), "FP_DELAY", np.mean(fp_delay))
    return tp, tn, fp, fn, np.mean(delay), np.mean(fp_delay)


def get_pareto_metrics_for_threshold(model, test_loader, batch_size, seq_len, threshold_list):
    fp_number_list = []
    fn_number_list = []
    delay_list = []
    fp_delay_list = []
    for threshold in threshold_list:
        tp, tn, fp_number, fn_number, mean_delay, mean_fp_delay = evaluate_metrics_on_set(model, test_loader,
                                                                                          batch_size, seq_len, threshold)

        fp_number_list.append(fp_number)
        fn_number_list.append(fn_number)
        delay_list.append(mean_delay)
        fp_delay_list.append(mean_fp_delay)

    return fp_number_list, fn_number_list, delay_list, fp_delay_list

def get_pareto_metrics_for_threshold_new(model, test_loader, batch_size, seq_len, threshold_list):
    fp_number_list = []
    fn_number_list = []
    delay_list = []
    fp_delay_list = []
    for threshold in threshold_list:
        tp, tn, fp_number, fn_number, mean_delay, mean_fp_delay = evaluate_metrics_on_set_new(model, test_loader,
                                                                                          batch_size, seq_len, threshold)

        fp_number_list.append(fp_number)
        fn_number_list.append(fn_number)
        delay_list.append(mean_delay)
        fp_delay_list.append(mean_fp_delay)

    return fp_number_list, fn_number_list, delay_list, fp_delay_list




def weird_division(n, d):
    return n / d if d else 0


def find_nearest_threshold(array, value, threshold_list):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    print(idx)
    return threshold_list[idx]


def cross_the_border(x_meaning, y_meaning, border):
    for x, y in zip(x_meaning, y_meaning):
        if x > border:
            return y


def model_crossing(model_name, model, test_loader, batch_size, seq_len, fp_delay, delay, threshold_list, x_coord):
    # x_coord = [40, 44, 48, 52, 56, 60, 62]
    y_coord = []
    thresold_value = []

    model_results_list = []

    for i in range(len(x_coord)):
        y_coord.append(cross_the_border(fp_delay, delay, x_coord[i]))
        thresold_value.append(find_nearest_threshold(delay, y_coord[i], threshold_list))
        true_positive, true_negative, false_positive, false_negative, _, _ = evaluate_metrics_on_set(model, test_loader,
                                                                                                     batch_size, seq_len,
                                                                                                     thresold_value[i])

        accuracy = weird_division((true_positive + true_negative),
                                  (true_positive + true_negative + false_positive + false_negative))
        precision = weird_division(true_positive, (true_positive + false_positive))
        recall = weird_division(true_positive, (true_positive + false_negative))
        f1_score = weird_division((2 * precision * recall), (precision + recall))
        spec = weird_division(true_negative, (false_positive + true_negative))
        g_mean = np.sqrt(recall + spec)
        curr_model_res_points = [model_name, x_coord[i], y_coord[i], thresold_value[i]]
        curr_mode_res_threshold = [true_positive, true_negative, false_positive, false_negative, accuracy, precision,
                                   recall, f1_score, g_mean]
        curr_model_res = curr_model_res_points + curr_mode_res_threshold

        model_results_list.append(curr_model_res)

    for i in range(len(x_coord)):
        print(model_name, "(Mean FP delay, Mean delay, Threshold): ", str([x_coord[i], y_coord[i]]),
              str(thresold_value[i]))
    print("---" * 10)

    return model_results_list

def area_under_graph(delay_list, fp_delay_list):
  return np.trapz(delay_list, fp_delay_list)

def save_metrics_to_file(model_name, model, test_loader, batch_size, seq_len, fp_delay, delay, x_coord, path_to_save = ""):
  columns_list = ["Model name", "Mean FP delay", "Mean delay", "Threshold", "TP", "TN", "FP", "FN", "Acc",
                "Precision", "Recall", "F1-score", "G-mean"]
  data_model_FNN_bce = model_crossing(model_name, model, test_loader, batch_size, seq_len, fp_delay, delay, x_coord)
  df_main = pd.DataFrame(data_model_FNN_bce, columns = columns_list)
  df_main.to_excel(os.path.join(path_to_save, model_name + ".xlsx" ), index = False)


# Правильный подсчет значений delay в выбранном fp_delay

def find_nearest_left_and_right_coord_index(array, value):
    array = np.asarray(array)
    left_idx = (np.abs(array - value)).argmin()
    if array[left_idx] > value:
        left_idx = left_idx - 1
    right_idx = left_idx + 1
    if right_idx > len(array):
        left_idx = left_idx - 1
        right_idx = right_idx - 1
    if left_idx < 0:
        left_idx = left_idx + 1
        right_idx = right_idx + 1
    return left_idx, right_idx

def lin_func(x1, y1, x2, y2, x):
    y = ((x - x1)*(y2 - y1)/(x2 - x1)) + y1
    return y

def find_threshold(fp_delay, x_coord, threshold_list):
    left_idx, right_idx = find_nearest_left_and_right_coord_index(fp_delay, x_coord)
    threshold = lin_func(fp_delay[left_idx], threshold_list[left_idx], fp_delay[right_idx], threshold_list[right_idx], x_coord)

    print(fp_delay[left_idx], threshold_list[left_idx], fp_delay[right_idx], threshold_list[right_idx], x_coord)

    return threshold


def metrics_by_fp_delay_x_coord(model_name, model, test_loader, batch_size, seq_len, fp_delay, threshold_list, x_coord_arr):
    y_coord_arr = []
    thresold_value = []

    model_results_list = []

    for x_coord in range(len(x_coord_arr)):
        threshold = find_threshold(fp_delay, x_coord_arr[x_coord], threshold_list)
        true_positive, true_negative, false_positive, \
        false_negative, delay, false_positive_delay = evaluate_metrics_on_set(model, test_loader, batch_size, seq_len, threshold)

        y_coord_arr.append(delay)
        thresold_value.append(threshold)

        accuracy = weird_division((true_positive + true_negative),
                                  (true_positive + true_negative + false_positive + false_negative))
        precision = weird_division(true_positive, (true_positive + false_positive))
        recall = weird_division(true_positive, (true_positive + false_negative))
        f1_score = weird_division((2 * precision * recall), (precision + recall))
        spec = weird_division(true_negative, (false_positive + true_negative))
        g_mean = np.sqrt(recall + spec)
        curr_model_res_points = [model_name, x_coord_arr[x_coord], delay, threshold]
        curr_model_res_threshold = [true_positive, true_negative, false_positive, false_negative, accuracy, precision,
                                   recall, f1_score, g_mean]
        curr_model_res = curr_model_res_points + curr_model_res_threshold

        model_results_list.append(curr_model_res)

    return model_results_list

# Функция принимает: название модели, модель, тест_лоадер, батч_сайз, массив fp_delay, массив интересущих нас значений
# fp_delay, путь куда сохранить таблицу . Значения fp_delay берутся при подсчете get_pareto_metrics_for_threshold.
# x_coord задаем сами. Данная версия отличается от предыдущей более правильным подсчетом trashold.
# Пример использования:
# fp_delay = [4, 8, 9, 12]
# x_coord_arr = [5, 10]
# save_metrics_to_file_new_version("ModelBCE",
#                                  model_bce, test_loader, BATCH_SIZE, fp_delay, x_coord_arr, path_to_save = "/home")
def save_metrics_to_file_new_version(model_name, model, test_loader, batch_size,
                                     seq_len, fp_delay, threshold_list, x_coord_arr, path_to_save = ""):
  columns_list = ["Model name", "Mean FP delay", "Mean delay", "Threshold", "TP", "TN", "FP", "FN", "Acc",
                "Precision", "Recall", "F1-score", "G-mean"]
  data_model_results = metrics_by_fp_delay_x_coord(model_name, model, test_loader, batch_size, seq_len,
                                                   fp_delay, threshold_list, x_coord_arr)
  df_main = pd.DataFrame(data_model_results, columns = columns_list)
  df_main.to_excel(os.path.join(path_to_save, model_name + ".xlsx" ), index = False)