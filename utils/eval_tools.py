import torch
import numpy as np
np.object = object # weird deprecation in current numpy version
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
from tqdm import tqdm

import forge
from forge import flags
import forge.experiment_tools as fet
from forge import load_from_checkpoint
from attrdict import AttrDict

from train_tools import param_count,nested_to
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from backbones_unet.model.losses import focal_loss

###########################################################################
# Binary classification functions and others

def get_acc(arr): # get accuracy of set of confusion matrices
	# arr.shape (n_thresholds,2,2)
	return np.array([(x[1,1] + x[0,0])/(x[0,0] + x[0,1] + x[1,0] + x[1,1]) if (x[0,0] + x[0,1] + x[1,0] + x[1,1])>0 else 0 for x in arr])

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

def moving_average(data, window_size):
	return uniform_filter1d(data, size=window_size, mode='nearest')

###########################################################################
# Main function for extracting metrics from a model run

def get_metrics(run_dir,n_thresholds =100,checkpoint_num = None,softmax = True,nonlin_thresholds = False,write_file = True,read_file = True):
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
	metrics_fname = osp.join(working_dir,'metrics.h5')

	if osp.exists(metrics_fname) and read_file: # check if metrics file exists and read metrics
		print("Found info file, reading from from file.")
		return dd.io.load(metrics_fname)
	
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
	kfold_loaders,k_inds = fet.load(data_config, config)
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

		for subset in conf_mats.keys(): # loop over train,test,val data subsets
			print("in subset",subset)
			loader = loader_dict[subset] # specific dataloader

			# evaluate model with batched data
			all_labels = []
			probs = []
			for dset in tqdm(loader):
				all_labels.append(dset['label'])
				batch_data = dset['data'].float()
				if config.include_coords:
					batch_coords = dset['coords'].float()
					model_input = {'data':batch_data,'coords':batch_coords}
				else:
					model_input = {'data':batch_data}#batch_data

				with torch.no_grad():
					if softmax:
						batch_probs = F.softmax(model(model_input),dim=1)
					else:
						batch_probs = model(model_input)
				probs.append(batch_probs)

			all_labels = torch.cat(all_labels,dim = 0)
			probs = np.array(torch.cat(probs,dim = 0))
			
			# all_labels = torch.tensor(loader.dataset.labels) # all predictions
			# # print('truelabels counts',np.bincount(all_labels))
			# all_data = torch.tensor(loader.dataset.data).float() # all inputs

			# if config.include_coords:
			#     model_input = {'data':all_data,'coords':torch.tensor(loader.dataset.coords).float()}
			# else:
			#     model_input = all_data
			# return model_input, model
			# with torch.no_grad():
			#     if softmax:
			#         print('attempting to evaluated mode')
			#         probs = np.array(F.softmax(model(model_input),dim=1))
			#     else:
			#         probs = np.array(model(model_input))

			print("model evaluated.")
			for ii,t in enumerate(thresholds): # for all classification thresholds,
				 # make binary predicitions at this threshold,
					
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
	
	results['avg_test_loss'] = (common_t,average_loss)
	
	# common_iter = 
	# results['avg_train_loss']
	
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

	if write_file:
		fname = osp.join(working_dir,'metrics.h5')
		print("Writing metrics to file",fname)
		dd.io.save(fname,results)
		
	print("Done.")
	return results

def get_image_metrics(model_config,data_config,experiment_name,config,n_thresholds = 100,mps = True,disable_tqdm = False):
	print("Working...")
	model,model_name = fet.load(model_config,config) # load model
	if mps:
		model = model.to('mps')
	
	kfold_loaders,k_inds = fet.load(data_config, config) # load data
	nfolds = len(kfold_loaders)

	conf_shape = (nfolds,n_thresholds,2,2)
	# list of confusion matrices (nfolds,n_thresholds)
	conf_mats = {'train':np.zeros(conf_shape),
				 'test':np.zeros(conf_shape)}

	thresholds = np.linspace(1e-8,1,n_thresholds)
	BCM = BinaryConfusionMatrix()

	results = {'thresholds':thresholds,'config':config,'params':param_count(model)}
	test_losses = []
	train_losses = []
	for k,loader_dict in enumerate(kfold_loaders):
		loss_file = osp.join('model_experiments',experiment_name,model_name + '_loss_fold{}.h5'.format(k+1))
		losses = dd.io.load(loss_file)
		test_losses.append(losses['test_loss'])
		train_losses.append(losses['train_loss'])

		wts_file = osp.join('model_experiments',experiment_name,model_name + '_wts_fold{}.pth'.format(k+1))
		model_wts = torch.load(wts_file)
		model.load_state_dict(model_wts)

		for subset in conf_mats.keys():
			loader = loader_dict[subset] # specific dataloader
			all_labels = []
			probs = []
			for (data,labels) in tqdm(loader,disable = disable_tqdm):
				labels = torch.flatten(labels,start_dim = 1)
				all_labels.append(labels)
				inputs = data.float()
				if mps: inputs = inputs.to('mps')
				preds = model.predict(inputs).cpu()
				batch_probs = torch.flatten(F.softmax(preds, dim = 1),start_dim = 2)
				probs.append(batch_probs)
			all_labels = torch.flatten(torch.cat(all_labels,dim = 0))
			probs = torch.flatten(torch.cat(probs,dim = 0).permute(1,0,2),start_dim = 1).permute(1,0)

			for ii,t in enumerate(thresholds): # for all classification thresholds,
					# make binary predicitions at this threshold,
					all_preds = np.zeros(len(probs),dtype = int)
					all_preds[probs[:,1] > t] = 1 # as threshold increases, fewer positives (FPR-> 0)
					all_preds = torch.tensor(all_preds)

					# compute confusion matrix
					binary_confusion_matrix = np.array(BCM(all_preds,all_labels),dtype = int)
					# record confusion matrix
					conf_mats[subset][k][ii] = binary_confusion_matrix # k is datafold index
	train_losses = np.array(train_losses)
	test_losses = np.array(test_losses)
	results['train_loss'] = np.mean(train_losses,axis = 0)
	results['test_loss'] = np.mean(test_losses,axis = 0)
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
	print("done.")
	return results

def quadPlot(elem,losses,model,test_loader,class_pred = 1,name = 'Model'): # plot everything
	fig, ax = plt.subplots(2, 2, figsize=(10, 10))  
	# plot loss 
	for case in ['train','test']:
		ax[0,0].plot(losses['{}_loss'.format(case)],label = case)
	ax[0,0].set_xlabel('epoch')
	ax[0,0].set_ylabel('focal loss')
	ax[0,0].set_yscale('log')
	ax[0,0].set_title(name + ' loss vs epochs')
	ax[0,0].legend()
	
	# plot spatial loss
	ex_data,ex_labels = next(iter(test_loader))
	preds = model.predict(ex_data).cpu()
	loss = focal_loss(preds,ex_labels,gamma = 10,alpha = 2,spatial = True)
	loss = loss.sum(dim = 1)
	im = ax[0,1].imshow(loss[elem])
	fig.colorbar(im, orientation='vertical')
	ax[0,1].axis('off')
	ax[0,1].set_title("Spatial loss on batch elem {}".format(elem))
	
	single_pred = preds[elem,...]
	single_true = ex_labels[elem,...]
	probs = torch.nn.functional.softmax(single_pred,dim = 0)
	# _, predicted_labels = torch.max(probs[:1], dim=0)
	predicted_labels = probs[class_pred,...]
	im3 = ax[1,0].imshow(single_true)
	ax[1,0].axis('off')
	ax[1,0].set_title("Ground truth")
	im2 = ax[1,1].imshow(predicted_labels,cmap = 'jet')  # You can change 'gray' to another colormap
	ax[1,1].set_title('Predictions (class = {})'.format(class_pred) ) # Set title for the first image
	ax[1,1].axis('off')
	fig.colorbar(im2)
	fig.colorbar(im3)

def getImgMetrics(model,train_loader,test_loader,n_thresholds = 1000,mps = False):
	# OLD FUNCTION BUT STILL WORKS FOR TESTING PURPOSES
	conf_shape = (n_thresholds,2,2)
	# list of confusion matrices (nfolds,n_thresholds)
	conf_mats = {'train':np.zeros(conf_shape),
				 'test':np.zeros(conf_shape)}
	if mps:
		model = model.to('mps')
	thresholds = np.linspace(1e-8,1,n_thresholds)
	BCM = BinaryConfusionMatrix()
	loader_dict = {'train':train_loader,'test':test_loader}
	results = {'thresholds':thresholds}
	for subset in conf_mats.keys():
		loader = loader_dict[subset] # specific dataloader
		all_labels = []
		probs = []
		for (data,labels) in tqdm(loader):
			labels = torch.flatten(labels,start_dim = 1)
			all_labels.append(labels)
			inputs = data.float()
			if mps: inputs = inputs.to('mps')
			preds = model.predict(inputs).cpu()
			batch_probs = torch.flatten(F.softmax(preds, dim = 1),start_dim = 2)
			# print('bp',batch_probs.shape)
			probs.append(batch_probs)
		all_labels = torch.flatten(torch.cat(all_labels,dim = 0))
		probs = torch.flatten(torch.cat(probs,dim = 0).permute(1,0,2),start_dim = 1).permute(1,0)

		for ii,t in enumerate(thresholds): # for all classification thresholds,
			 # make binary predicitions at this threshold,    
			all_preds = np.zeros(len(probs),dtype = int)
			all_preds[probs[:,1] > t] = 1 # as threshold increases, fewer positives (FPR-> 0)
			all_preds = torch.tensor(all_preds)
			# compute confusion matrix
			binary_confusion_matrix = np.array(BCM(all_preds,all_labels),dtype = int)
			# record confusion matrix
			conf_mats[subset][ii] = binary_confusion_matrix
			
	for subset in conf_mats.keys(): # for train, test, val
		thresh_mats = conf_mats[subset] # sum confusion matrices over data folds
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

	return results

def metricsPlot(run_info,model_name = None,savefile = False):
	fig, ax = plt.subplots(2, 2, figsize=(10, 10))
	fig.suptitle("Model: {} classifier metrics".format(model_name))
	# plot accuracy
	ts = run_info['thresholds']
	subsets = ['train','test']
	for subset in subsets:
		accs = run_info[subset+'_acc']
		ax[0,0].plot(ts,accs,label = subset)
	ax[0,0].set_xlabel("classification threshold")
	ax[0,0].set_ylabel("accuracy")
	ax[0,0].set_title("Classifier accuracy vs. threshold")
	ax[0,0].legend()
	# plot F1
	for subset in subsets:
		accs = run_info[subset+'_F1']
		thresh_max = ts[np.argmax(accs)]
		ax[0,1].plot(ts,accs,label = subset,)
	ax[0,1].set_xlabel("classification threshold")
	ax[0,1].set_ylabel("F1 Score")
	ax[0,1].set_title("F1 max at threshold = {}".format(np.round(thresh_max,decimals = 3)))
	ax[0,1].legend()
	# plot ROC
	for subset in subsets:
		tprs = np.append(run_info[subset+'_tpr'][::-1],1)
		fprs = np.append(run_info[subset+'_fpr'][::-1],1)
		auc = np.trapz(tprs,fprs)
		ax[1,0].plot(fprs,tprs,label = subset + "; AUC = {}".format(np.round(auc,decimals = 4)))
	line = np.linspace(0,1,100)
	ax[1,0].plot(line,line,'k--')
	ax[1,0].set_xlabel("False positive rate")
	ax[1,0].set_ylabel("True positive rate")
	ax[1,0].set_title("ROC Curves; in/out of sample")
	ax[1,0].legend()
	for subset in subsets:
		precs = run_info[subset+'_prec'][::-1]
		recs = run_info[subset+'_tpr'][::-1]
		ax[1,1].plot(recs,precs,label = subset)
	ax[1,1].set_xlabel("Recall")
	ax[1,1].set_ylabel("Precision")
	ax[1,1].set_title("PR Curves; out of sample")
	ax[1,1].legend()
	plt.show()

	if savefile:
		fig.savefig(osp.join('comparison_images',model_name+'_metrics.pdf'),bbox_inches = 'tight')

###########################################################################
###########################################################################
# Plotting various metrics


# PLOT TEST LOSS VS TIME
def plot_test_loss(run_info,**kwargs):
	times = run_info['test_time']
	losses = run_info['test_loss']
	ts,avg = run_info['avg_test_loss']
	for k in range(len(times)):
		plt.plot(times[k],losses[k],label = 'fold {} loss'.format(k+1))
	plt.plot(ts,avg,'k--',linewidth = 2,label = 'avg loss')
	plt.xlabel("training time")
	# plt.xscale('log')
	plt.ylabel("test loss")
	plt.title("Test loss vs train time")
	plt.legend()
	plt.show()

def plot_train_test_loss(run_info):
	ts = run_info['train_iter']
	ls = run_info['train_loss']
	tts,avg = run_info['avg_test_loss']
	
	plt.plot(tts,avg,linewidth = 2,label = 'avg test loss')
	min_time = min([np.min(t) for t in ts])
	max_time = max([np.max(t) for t in ts])
	ts_common = np.linspace(0, 1, 1000)
	interpolated_arrays = []
	cutoff = 10
	for i,(t, l) in enumerate(zip(ts, ls)):
		t = t[cutoff:]/max(t)
		l = l[cutoff:]
		smoothed_l = moving_average(l, 30)
		interp_func = interp1d(t, smoothed_l, bounds_error=False, fill_value=(smoothed_l[0], smoothed_l[-1]))
		interpolated_arrays.append(interp_func(ts_common))
	
	interpolated_arrays = np.array(interpolated_arrays)
	ls_common = np.nanmean(interpolated_arrays, axis=0)
	
	plt.plot(ts_common[cutoff:],ls_common[cutoff:],linewidth = 2,label = "avg train loss")
	plt.xlabel("progress fraction")
	plt.ylabel("Cross-entropy loss")
	plt.title("Test vs train loss")
	plt.legend()

# PLOT TRAIN LOSS VS ITER
def plot_train_loss(run_info):
	ts = run_info['train_iter']
	ls = run_info['train_loss']
	min_time = min([np.min(t) for t in ts])
	max_time = max([np.max(t) for t in ts])
	ts_common = np.linspace(min_time, max_time, 1000)
	interpolated_arrays = []
	cutoff = 10
	for i,(t, l) in enumerate(zip(ts, ls)):
		t = t[cutoff:]
		l = l[cutoff:]
		smoothed_l = moving_average(l, 50)
		plt.plot(t,smoothed_l,label = "fold {}".format(i+1))
		interp_func = interp1d(t, smoothed_l, bounds_error=False, fill_value=(smoothed_l[0], smoothed_l[-1]))
		interpolated_arrays.append(interp_func(ts_common))
	
	interpolated_arrays = np.array(interpolated_arrays)
	ls_common = np.nanmean(interpolated_arrays, axis=0)
	
	plt.plot(ts_common[cutoff:],ls_common[cutoff:], 'k--',linewidth = 2,label = "avg loss")
	plt.xlabel("training iteration")
	plt.ylabel("training loss")
	plt.title("Training loss vs. iteration")
	plt.legend()

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
def plot_F1(run_info,get_max = True,**kwargs):
	ts = run_info['thresholds']
	subsets = ['test']
	for subset in subsets:
		accs = run_info[subset+'_F1']
		thresh_max = ts[np.argmax(accs)]
		plt.plot(ts,accs,label = subset,**kwargs)
	plt.xlabel("classification threshold")
	plt.ylabel("F1 Score")
	plt.title("F1 score vs. threshold")
	plt.legend()
	plt.show()
	print("F1 score maximized at threshold",np.round(thresh_max,decimals = 3))

# PLOT ROC CURVES
def plot_ROC(run_info,**kwargs):
	subsets = ['train','test','val']
	for subset in subsets:
		tprs = np.append(run_info[subset+'_tpr'][::-1],1)
		fprs = np.append(run_info[subset+'_fpr'][::-1],1)
		auc = np.trapz(tprs,fprs)
		plt.plot(fprs,tprs,label = subset + "; AUC = {}".format(np.round(auc,decimals = 4)),**kwargs)
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