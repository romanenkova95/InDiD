"""Module with attention masks for network models with transformer"""

import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
import random
import numpy as np
from CPD.new_metrics_2 import * #as new_metrics
from CPD import new_metrics
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
            model.to(device)
            model.model.device='cpu'
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
    try:
        exp_name = model.model.experiment_name
    except:
        exp_name = ''
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
    elif dataset_name[:9] == 'synthetic':
                title += 'synthetic 1D' + ', seq len '+ str(SEQ_LEN)+', '+ model.loss_type
    else:
        print('Error: ', dataset_name)
        return -2
    print('\n')
    print(title)
    f, axs = plt.subplots(n_rows, n_cols, sharey=True)
    f.set_figheight(15)
    f.set_figwidth(15)
    f.suptitle(title)
    try:
        for i in range(n_rows):
            n, j = 0, 0
            for inputs, labels in model.val_dataloader():
                n+=1
                if n > random.randint(0, len(model.val_dataloader())):
                    try:
                        pred = model(inputs.to(device).float()).squeeze(2)
                    except:
                        pred = model(inputs.to(device).long()).squeeze(2)
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
                try:
                    file_name = model.model.experiment_name
                except:
                    file_name = 'pos_enc'
            if dataset_name[-7:] == 'changes':
                path_ = './n_changes_experiment/'
            elif dataset_name[:5] == 'mnist':
                path_ = './seqlen_experiment/'
            elif dataset_name[:9] == 'synthetic':
                    path_ = '.synthetic/'
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
    except Exception as e:
        print(e, '\n')

# Calculate pareto_metrics_for_threshold, cover, F1_score_ruprures, F1_score for 0.2, 0.5, 0.7
def get_model_metrics(train_model,
                      n, 
                      exp_type,
                      exp_name,
                      dataset_name, 
                      SEQ_LEN=64,
                      threshold_number = 50, 
                      without_mask=False,
                      savefigure=False, 
                      savetext=False):
        try:
            if next(train_model.model.parameters()).is_cuda :
                device = 'cuda'
            else:
                device = 'cpu'
            train_model.model.device = device
            train_model.to(device)
            train_model.model.to(device)
            if without_mask == False:
                train_model.model.src_mask = train_model.model.src_mask.to(device)
             
        except:
            print("Something wrong with devices")
        
        threshold_list = np.linspace(-15, 15, threshold_number)
        threshold_list = 1 / (1 + np.exp(-threshold_list))
        threshold_list = [-0.001] + list(threshold_list) + [1.001]
        model_metrics = []
        ##############################################################################################################
        filename = exp_name + '_v' + str(n) + '_' + exp_type
                
        try:
            pos_enc = train_model.model.pos_enc
        except:
            pos_enc = ''
        model_metrics.append(str(SEQ_LEN) + '_' + filename + '_' + dataset_name + '_' + pos_enc)
        
        
        ##############################################################################################################
        
        
        ############################################################################################################        
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
            ) = evaluate_metrics_on_set(model=train_model, test_loader=train_model.test_dataloader(), threshold=threshold, 
                                        verbose=False, model_type='seq2seq', device=device)

            #confusion_matrix_dict[threshold] = (TN, FP, FN, TP)
            delay_dict[threshold] = mean_delay
            fp_delay_dict[threshold] = mean_fp_delay        

            #cover_dict[threshold] = cover
            #f1_dict[threshold] = F1_score((TN, FP, FN, TP))
            
        auc = area_under_graph(list(delay_dict.values()), list(fp_delay_dict.values()))   
        model_metrics.append(auc)
        
        # Conf matrix and F1
        #best_th_f1 = max(f1_dict, key=f1_dict.get)

        #best_conf_matrix = (confusion_matrix_dict[best_th_f1][0], confusion_matrix_dict[best_th_f1][1], 
        #                    confusion_matrix_dict[best_th_f1][2], confusion_matrix_dict[best_th_f1][3])
        #best_f1 = f1_dict[best_th_f1]
        
        # Cover
        #best_cover = cover_dict[best_th_f1]

        #best_th_cover = max(cover_dict, key=cover_dict.get)
        #max_cover = cover_dict[best_th_cover]
        #model_metrics.append(best_f1)
        #model_metrics.append(best_cover)
        #model_metrics.append(max_cover)
        
        # thresholds
        ris, hs, f1rs = rupture_pipeline(train_model, device)
        covers, f1s = F1_pipeline(train_model, [0.2, 0.5, 0.7], device )
        for t in [0.2, 0.5, 0.7]:
            model_metrics.append(np.mean(covers[t]))
                                 
            model_metrics.append(np.mean(f1rs[t]))
            model_metrics.append(np.mean(f1s[t]))
            model_metrics.append(np.mean(ris[t]) )
            model_metrics.append(np.mean(hs[t]))
            
       
        if savetext:
            if dataset_name[-7:] == 'changes':
                file_metrics = 'n_changes_experiment/new_all_metrics.txt'
            elif dataset_name[:5] == 'mnist':
                file_metrics = 'seqlen_experiment/data2_mask_all_metrics.txt'
            elif dataset_name == 'wiki':
                file_metrics = str(SEQ_LEN)+'_text_1layer_all_metrics.txt'
            with open(file_metrics, 'a') as f:
                print('Save metrics in ', file_metrics)
                f.writelines( str(model_metrics)+', \n')
                
        return model_metrics
