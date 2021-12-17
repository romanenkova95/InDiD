"""Module with attention masks for network models with transformer"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
from CPD import metrics, new_metrics
import os
from numpy import asarray
from numpy import savetxt

# NEW
# drawing 6 prediction's plots
def plot_pred(model, SEQ_LEN, dataset_name, n_rows=2, n_cols=3, without_mask=False,  savefigure=False):
    try:
        if next(model.model.parameters()).is_cuda :
            device = 'cuda'
            if not next(model.parameters()).is_cuda :
                model.to('cuda')
        else:
            device = 'cpu'
    except:
        device = 'cpu'
   
    print("Device in prediction's plots: ", device)
    try:
        for i in range(len(model.model.masks)):
            model.model.masks[i] = model.model.masks[i].to(device)
        print("All masks are on ", device)
    except:
        print('Not module transformer_3masks')
    try:
        model.model.src_mask = model.model.src_mask.to(device)
        print("Src mask is on ", device)
    except:
        print('Not module transformer with mask')
    
    exp_name = model.model.experiment_name
    try:
        if model.model.name_mask != 'without':
            title = exp_name + ' with ' + model.model.name_mask + ' mask, '
        else:
            title = exp_name + model.model.name_mask + ' mask, '
    except:
        title = exp_name
    if dataset_name[-7:] == 'changes':
        
            title += ' ' +dataset_name+', '+ model.loss_type
    elif dataset_name[:5] == 'mnist':
            title += ', seq len '+ str(SEQ_LEN)+', '+ model.loss_type
    else:
        print('Error: ', dataset_name)
        return -2
    print(title)
    f, axs = plt.subplots(n_rows, n_cols, sharey=True)
    f.set_figheight(15)
    f.set_figwidth(15)
    f.suptitle(title)
    for i in range(n_rows):
        n, j = 0, 0
        for inputs, labels in model.val_dataloader():
            n+=1
            if n > random.randint(0, len(model.val_dataloader())):
                pred = model(inputs.to(device)).squeeze(2)
                for j in range(n_cols):
                    k = random.randint(0, len(pred - 1))
                    axs[i][j].plot(pred[k].detach().cpu().numpy(), label='pred')
                    axs[i][j].plot(labels[k].detach().cpu().numpy(), label='true')
                    axs[i][j].legend()
                    
                break
                    #j += 1
    f.show()
    if savefigure:
        try:
            file_name = model.model.name_mask  + '_mask'
        except:
            file_name = model.model.experiment_name
        if dataset_name[-7:] == 'changes':
            path_ = './n_changes_experiment/'
        if dataset_name[:5] == 'mnist':
            path_ = './seqlen_experiment/'
        try:
            if not os.path.exists(os.path.dirname(path_+'img/')):
                os.makedirs(os.path.dirname(path_+'img/'))
            if not os.path.exists(os.path.dirname(path_+'img/'+model.loss_type.lower()+'/')):
                os.makedirs(os.path.dirname(path_+'img/'+model.loss_type.lower()+'/'))
            if not os.path.exists(os.path.dirname(path_+'img/'+model.loss_type.lower()+'/'+ str(SEQ_LEN) + '/')):
                os.makedirs(os.path.dirname(path_+'img/'+model.loss_type.lower()+'/'+ str(SEQ_LEN) + '/'))
        except OSError as err:
            print(err)
            
        plt.savefig(path_+'img/'+model.loss_type.lower()+'/'+ str(SEQ_LEN) + '/' + file_name +'.png')

# Calculate pareto_metrics_for_threshold, cover, F1_score_ruprures, F1_score for 0.2, 0.5, 0.7
def get_model_metrics(train_model,
                      base_model,
                      n, 
                      exp_type,
                      exp_name,
                      dataset_name, 
                      SEQ_LEN=64,
                      threshold_number = 50, 
                      without_mask=False,
                      savefigure=False, 
                      savetext=False):
    
        threshold_list = np.linspace(-15, 15, threshold_number)
        threshold_list = 1 / (1 + np.exp(-threshold_list))
        threshold_list = [-0.001] + list(threshold_list) + [1.001]
        
        model_metrics = []
        
        plot_pred(model=train_model,  SEQ_LEN=SEQ_LEN, dataset_name=dataset_name, #file_name=exp_name+'_v'+str(n)+'_'+exp_type, 
                #title='3 masks, CPD loss, seq_len = '+str(SEQ_LEN),
                 without_mask = without_mask, savefigure=savefigure)  
        (_, _, delay_list, fp_delay_list) = metrics.get_pareto_metrics_for_threshold(
                                                                                train_model, train_model.test_dataloader(),
                                                                                threshold_list,
                                                                                device=train_model.device.type
                                                                                    )
        filename = exp_name + '_v' + str(n) + '_' + exp_type
        if savetext:
            if dataset_name[-7:] == 'changes':
                path_ = 'n_changes_experiment/'+ dataset_name + '/area_plots/'
            elif dataset_name[:5] == 'mnist':
                path_ = 'seqlen_experiment/'+str(SEQ_LEN)+'/area_plots/'
            try:
                if not os.path.exists(os.path.dirname('./'+path_)):
                    os.makedirs(os.path.dirname('./'+path_))
            except OSError as err:
                print(err)
            
            savetxt(path_ + 'delay_list_'    + filename + '.csv', delay_list, delimiter=',')
            savetxt(path_ + 'fp_delay_list_' + filename + '.csv', fp_delay_list, delimiter=',')
            
        model_metrics.append(str(SEQ_LEN) + '_' + filename + '_' + dataset_name )
        model_metrics.append(new_metrics.area_under_graph(delay_list, fp_delay_list))
        # thresholds
        for t in [0.2, 0.5, 0.7]:
            model_metrics.append(new_metrics.cover(train_model, train_model.test_dataloader(), t))
            model_metrics.append(new_metrics.F1_score_ruptures(train_model, train_model.test_dataloader(), t))
            model_metrics.append(new_metrics.F1_score(train_model, train_model.test_dataloader(), t))
        if savetext:
            if dataset_name[-7:] == 'changes':
                file_metrics = 'n_changes_experiment/all_metrics.txt'
            elif dataset_name[:5] == 'mnist':
                file_metrics = 'seqlen_experiment/all_metrics.txt'
            with open(file_metrics, 'a') as f:
                print('Save metrics in ', file_metrics)
                f.writelines( str(model_metrics)+', \n')
                
        return model_metrics
