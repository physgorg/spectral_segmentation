import torch
import numpy as np
import json
import torch.nn.functional as F
import deepdish as dd

import torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix

import matplotlib.pyplot as plt
from os import path as osp
import os
import re
import h5py

import forge
from forge import flags
import forge.experiment_tools as fet
from forge import load_from_checkpoint
from attrdict import AttrDict

from train_tools import param_count,nested_to

###########################################################################
# Binary classification functions

def get_acc(arr): # get accuracy of set of confusion matrices
    # arr.shape (n_thresholds,2,2)
    return np.array([(x[1,1] + x[0,0])/(x[0,0] + x[0,1] + x[1,0] + x[1,1]) if (x[0,0] + x[0,1] + x[1,0] + x[1,1]) else 0 for x in arr])

def get_prec(arr): # get precision of set of confusion matrices
    # arr.shape (n_thresholds,2,2)
    return np.array([(x[1,1])/(x[1,1] + x[0,1]) if (x[1,1] + x[0,1]) > 0 else 0 for x in arr])

def get_tpr(arr): # get true positive rate/recall of set of confusion matrices
    # arr.shape (n_thresholds,2,2)
    return np.array([(x[1,1])/(x[1,1] + x[1,0]) if (x[1,1] + x[1,0]) > 0 else 0 for x in arr])

def get_fpr(arr): # get false positive rate of set of confusion matrices
    # arr.shape (n_thresholds,2,2)
    return np.array([(x[0,1])/(x[0,1] + x[0,0]) if (x[0,1] + x[0,0]) > 0 else 0 for x in arr])

def get_F1(arr): # get F1-score of set of confusion matrices
    # arr.shape (n_thresholds,2,2)
    return np.array([2*x[1,1]/(2*x[1,1] + x[0,1] + x[1,0]) if (2*x[1,1] + x[0,1] + x[1,0]) > 0 else 0 for x in arr])


###########################################################################
# Main function for extracting metrics from a model run

def get_metrics(run_dir,n_thresholds =100,checkpoint_num = None,softmax = True,nonlin_thresholds = False):
    # run directory should contain folders 1,2,3... 
    if nonlin_thresholds:
        num_elements = n_thresholds

        # Create a non-linear space with more density at lower values
        # Using exponential space for higher resolution at smaller thresholds
        # Adjust the base of the exponential to control the distribution
        non_linear_part = np.exp(np.linspace(-np.log(1000), 0, num_elements // 2))
        
        # Create a linear space for the higher thresholds
        linear_part = np.linspace(0, 1, num_elements // 2)
        
        # Combine both parts
        thresholds = np.unique(np.concatenate([non_linear_part, linear_part]))
        
        # Ensure the thresholds are sorted (necessary if combining different spaces)
        thresholds.sort()
    else:
        thresholds = np.linspace(1e-8,1,n_thresholds)
    BCM = BinaryConfusionMatrix()
    ###########
    # Note: BCM = [[TN,FP],[FN,TP]] 
    
    if checkpoint_num is None: # if None, get most recent directory
        checkpoint_num = max([int(x) for x in os.listdir(run_dir) if osp.isdir(os.path.join(run_dir, x))])

    working_dir = osp.join(run_dir,str(checkpoint_num))
    print("Getting metrics from",working_dir)
    flag_file = osp.join(working_dir,'flags.json')
    with open(flag_file, 'r') as file:
        flag_info = json.load(file)
    config = AttrDict(flag_info)

    model_config = osp.join(working_dir,config['model_config'])
    data_config = osp.join(working_dir,config['data_config'])

    # print(model_config)
    # print(data_config)

    # load model (same for all kfolds)
    model,model_name = fet.load(model_config,config)

    # load data
    kfold_loaders = fet.load(data_config, config)
    nfolds = len(kfold_loaders)
    print("Data is {}-fold partitioned.".format(nfolds))

    conf_shape = (nfolds,n_thresholds,2,2)
    # list of confusion matrices (nfolds,n_thresholds)
    conf_mats = {'train':np.zeros(conf_shape),
                 'test':np.zeros(conf_shape),
                 'val':np.zeros(conf_shape)} 

    results = {'thresholds':thresholds,'config':config,'params':param_count(model)}

    test_times = []
    test_losses = []
    train_iters = []
    train_losses = []
    # val_times = [] # these are mostly redundant for now
    # val_losses = []

    ###########################################
    # RECORD METRICS, CONFUSION MATRICES FOR ALL THRESHOLDS FOR ALL FOLDS, MODELS
    
    for k,loader_dict in enumerate(kfold_loaders):
        print("Evaluating fold {}...".format(k+1))
        fold_dir = osp.join(working_dir,"data_fold{}".format(k+1))

        # record times and losses from results_dict.h5
        test_reports = dd.io.load(osp.join(fold_dir,'results_dict.h5'))
        test_times.append(test_reports['time']/max(test_reports['time']))
        test_losses.append(test_reports['cross_entropy'])

        iters,losses = np.array(dd.io.load(osp.join(fold_dir,'results_dict_train.h5'))).T
        train_iters.append(iters)
        train_losses.append(losses)
        # val_reports = dd.io.load(osp.join(fold_dir,'results_dict_val.h5'))
        # val_times.append(val_reports['time'])
        # val_losses.append(val_reports['cross_entropy']
        
        # find and load latest model checkpoint in fold dir
        model_ckpt_name = "model_fold{}.ckpt-".format(k+1)
        ckpt_files = [f for f in os.listdir(fold_dir) if f.startswith(model_ckpt_name) and re.search(r'\d+$', f)]
        latest_ckpt = max(ckpt_files, key=lambda x: int(re.search(r'(\d+)$', x).group()), default=None)

        ckpt_file = osp.join(fold_dir, latest_ckpt)
        
        # load model weights from final checkpoint and set to evaluation mode
        model_wts = torch.load(ckpt_file)

        model.load_state_dict(model_wts['model_state_dict'])
        model.eval()

        # set up model to make binary int class predictions from logit output
        # def model_preds(x,threshold = 0.5,softmax = True):
        #     if softmax:
        #         probs = np.array(F.softmax(model(x),dim=1))
        #     else:
        #         probs = np.array(model(x))
        #     print(probs[:,1])
        #     preds = np.zeros(len(probs),dtype = int)
        #     preds[probs[:,1] > threshold] = 1 # CONFUSED ABOUT DIRECTION OF THIS INEQ
        #     return torch.tensor(preds)

        for subset in conf_mats.keys(): # loop over train,test,val data subsets
            print("in subset",subset)
            loader = loader_dict[subset] # specific dataloader
            
            all_labels = torch.tensor(loader.dataset.labels) # all predictions
            # print('truelabels counts',np.bincount(all_labels))
            all_data = torch.tensor(loader.dataset.data).float() # all inputs

            with torch.no_grad():
                if softmax:
                    probs = np.array(F.softmax(model(all_data),dim=1))
                else:
                    probs = np.array(model(all_data))

            for ii,t in enumerate(thresholds): # for all classification thresholds,
                 # make binary predicitions at this threshold,
                    # print('threshold',t)
                    # all_preds = model_preds(all_data,threshold = t)
                all_preds = np.zeros(len(probs),dtype = int)
                
                all_preds[probs[:,1] > t] = 1 # as threshold increases, fewer positives (FPR-> 0)

                all_preds = torch.tensor(all_preds)

                # compute confusion matrix
                binary_confusion_matrix = np.array(BCM(all_preds,all_labels),dtype = int)

                # record confusion matrix
                conf_mats[subset][k][ii] = binary_confusion_matrix

    ##########################################################
    # AGGREGATE PREDICTIONS OVER DATA FOLDS AND RETURN METRICS

    # collect loss vs. time data from training
    results['test_time'] = test_times
    results['test_loss'] = test_losses

    results['train_iter'] = train_iters
    results['train_loss'] = train_losses
    
    common_t = np.linspace(0, 1, 100)
    interpolated_y_values = np.array([interp1d(x, y, bounds_error=False, fill_value='extrapolate')(common_t) for x, y in zip(test_times, test_losses)])
    average_loss = np.mean(interpolated_y_values, axis=0)
    
    results['avg_loss'] = (common_t,average_loss)
    
    print("Models evaluated. Computing metrics...")
    for subset in conf_mats.keys(): # for train, test, val
        thresh_mats = np.sum(conf_mats[subset],axis = 0) # sum confusion matrices over data folds
        
        # compute accuracy
        results[subset+'_acc'] = get_acc(thresh_mats)
        # compute precision
        results[subset+'_prec'] = get_prec(thresh_mats)
        # compute TPR
        results[subset+'_tpr'] = get_tpr(thresh_mats)
        # compute FPR
        results[subset+'_fpr'] = get_fpr(thresh_mats)
        # compute F1 score
        results[subset+'_F1'] = get_F1(thresh_mats)
    print("Done.")
    return results

###########################################################################
###########################################################################
# Plotting various metrics


# PLOT TEST LOSS VS TIME
def plot_loss(run_info,**kwargs):
    times = run_info['test_time']
    losses = run_info['test_loss']
    ts,avg = run_info['avg_loss']
    for k in range(len(times)):
        plt.plot(times[k],losses[k],label = 'fold {} loss'.format(k+1))
    plt.plot(ts,avg,'k--',linewidth = 2,label = 'avg loss')
    plt.xlabel("training time")
    # plt.xscale('log')
    plt.ylabel("test loss")
    plt.legend()
    plt.show()

# PLOT ACCURACY VS THRESHOLD
def plot_accuracy(run_info,**kwargs):
    ts = run_info['thresholds']
    subsets = ['train','test','val']
    for subset in subsets:
        accs = run_info[subset+'_acc']
        plt.plot(ts,accs,label = subset,**kwargs)
    plt.xlabel("classification threshold")
    plt.ylabel("accuracy")
    plt.title("Classifier accuracy vs. threshold")
    plt.legend()
    plt.show()

# PLOT F1 SCORE VS THRESHOLD
def plot_F1(run_info,**kwargs):
    ts = run_info['thresholds']
    subsets = ['test']
    for subset in subsets:
        accs = run_info[subset+'_F1']
        plt.plot(ts,accs,label = subset,**kwargs)
    plt.xlabel("classification threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 score vs. threshold")
    plt.legend()
    plt.show()

# PLOT ROC CURVES
def plot_ROC(run_info,**kwargs):
    subsets = ['train','test','val']
    for subset in subsets:
        tprs = np.append(run_info[subset+'_tpr'][::-1],1)
        fprs = np.append(run_info[subset+'_fpr'][::-1],1)
        auc = np.trapz(tprs,fprs)
        plt.plot(fprs,tprs,label = subset + "; AUC = {}".format(auc),**kwargs)
    line = np.linspace(0,1,100)
    plt.plot(line,line,'k--')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curves; in/out of sample")
    plt.legend()
    plt.show()

# PLOT PR CURVES
def plot_PR(run_info,**kwargs):
    subsets = ['test','val']
    for subset in subsets:
        precs = run_info[subset+'_prec'][::-1]
        recs = run_info[subset+'_tpr'][::-1]
        plt.plot(recs,precs,label = subset,**kwargs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curves; out of sample")
    plt.legend()
    plt.show()