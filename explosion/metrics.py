import torch
import numpy as np


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
    # KOSTUL

    delay = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    true_positive = 0

    index_real = torch.where(y_true != y_true[0])[0] # indexes with real changes
    index_detected = torch.where((y_pred > threshold).long() != y_true[0])[0] # indexes with predicted changes

    if len(index_real) > 0:
        real_change_index = index_real[0]

        if len(index_detected) > 0:
            index_sub_detected = torch.where(index_detected >= real_change_index)[0]

            if len(index_sub_detected) == len(index_detected):
                false_positive_delay = real_change_index.item()
                detected_change_index = index_detected[0]
                delay = (detected_change_index - real_change_index).item()
                true_positive +=1
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
    
    accuracy = ((y_pred > threshold).long() == y_true).float().mean().item()

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
    # KOSTUL

    delay = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    true_positive = 0

    index_real = torch.where(y_true != y_true[0])[0] # indexes with real changes
    index_detected = torch.where((y_pred > threshold).long() != y_true[0])[0] # indexes with predicted changes

    if len(index_real) > 0:
        real_change_index = index_real[0]

        if len(index_detected) > 0:
            index_sub_detected = torch.where(index_detected >= real_change_index)[0]

            if len(index_sub_detected) == len(index_detected):
                false_positive_delay = seq_len  # New change here 
                detected_change_index = index_detected[0]
                delay = (detected_change_index - real_change_index).item()
                true_positive +=1
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
    
    accuracy = ((y_pred > threshold).long() == y_true).float().mean().item()

    return true_positive, true_negative, false_negative, false_positive, delay, false_positive_delay

def evaluate_metrics_on_set(model, test_loader, batch_size, seq_len, threshold = 0.5, device='cuda'):
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
      #print(val_out)
      # test_loss = loss_function(test_out.squeeze(), test_labels.float().squeeze())
      # test_losses.append(test_loss.item())
      test_outputs += [test_out.cpu().clone().detach()]
      
      for l, o in zip(test_labels.squeeze(), test_out.squeeze()):

        tp_cur, tn_cur, fn_cur, fp_cur, delay_curr, fp_delay_curr = evaluate_metrics(l, o, seq_len, threshold)

        tp += tp_cur
        fp += fp_cur
        tn += tn_cur
        fn += fn_cur

        #print(test_labels.squeeze(), test_out.squeeze())

        delay.append(delay_curr)
        fp_delay.append(fp_delay_curr)
  
  test_outputs_one = torch.cat(test_outputs, 0)

  overall_test_loss_list.append(np.mean(test_losses))

  print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn, "DELAY:", np.mean(delay), "FP_DELAY", np.mean(fp_delay))
  # print(overall_test_loss_list)

  return tp, tn, fp, fn, np.mean(delay), np.mean(fp_delay)

def evaluate_metrics_on_set_new(model, test_loader, batch_size, seq_len, threshold = 0.5, device='cuda'):
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
      #print(val_out)
      # test_loss = loss_function(test_out.squeeze(), test_labels.float().squeeze())
      # test_losses.append(test_loss.item())
      test_outputs += [test_out.cpu().clone().detach()]
      
      for l, o in zip(test_labels.squeeze(), test_out.squeeze()):

        tp_cur, tn_cur, fn_cur, fp_cur, delay_curr, fp_delay_curr = evaluate_metrics_new(l, o, seq_len, threshold)

        tp += tp_cur
        fp += fp_cur
        tn += tn_cur
        fn += fn_cur

        # print(labels.squeeze(), output.squeeze())

        delay.append(delay_curr)
        fp_delay.append(fp_delay_curr)
  
  test_outputs_one = torch.cat(test_outputs, 0)

  overall_test_loss_list.append(np.mean(test_losses))

  print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn, "DELAY:", np.mean(delay), "FP_DELAY", np.mean(fp_delay))
  # print(overall_test_loss_list)

  return tp, tn, fp, fn, np.mean(delay), np.mean(fp_delay)
  
def get_pareto_metrics_for_threshold(model, test_loader, batch_size, threshold_list, seq_len):
    fp_number_list = []
    fn_number_list = []
    delay_list = []
    fp_delay_list = []
    for threshold in threshold_list:
        # (positive_number, negative_number, test_loss, 
        #  test_acc, mean_delay, mean_fp_delay, fp_number, fn_number) = get_quality_metrics(model, test_loader, threshold)
        tp, tn, fp_number, fn_number, mean_delay, mean_fp_delay = evaluate_metrics_on_set(model, test_loader, batch_size, seq_len, threshold)

        fp_number_list.append(fp_number)
        fn_number_list.append(fn_number)
        delay_list.append(mean_delay)
        fp_delay_list.append(mean_fp_delay)
    
    return fp_number_list, fn_number_list, delay_list, fp_delay_list
