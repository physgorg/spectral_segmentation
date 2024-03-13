import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import RepeatedKFold

from backbones_unet.model.unet import Unet
from backbones_unet.model.losses import DiceLoss,focal_loss,FocalLoss
from backbones_unet.utils.trainer import Trainer

from data_configs.penn_image_dataset import PennImageData,PennImageDataU
import utils.eval_tools as evl

import os
from os import path as osp
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from attrdict import AttrDict
import deepdish as dd

import forge
from forge import flags
import forge.experiment_tools as fet

from train_tools import param_count,nested_to
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba_array
import matplotlib


HEADINGS = np.array(['aaa0043','aaa0044','aaa0060','aaa0061','aaa0063','aaa0064','aaa0066','aaa0069','aaa0086','aaa0087','aaa_0051','aaa_0053','aaa_0054','aaa_0059','aaa_0071','aaa_0072'])

class PennData(Dataset): # importing here to avoid forge flags conflicts

	def __init__(self,path_to_data,headings,k_inds = None,include_coords = True,rescale = True,spectra_prefix = "kvals_fuse_rotate_",label_prefix = 'labels_fuse_rotate_'):
		
		self.data_dir = path_to_data
		self.headings = headings
		self.inclx = include_coords

		df_list = []
		label_list = []
		for idx,heading in enumerate(self.headings): # for each datafile heading,
			data_name = osp.join(self.data_dir,spectra_prefix + heading+'.csv') # read kvals
			df_temp = pd.read_csv(data_name,header=None)
			df_temp.insert(0, 'h_idx', idx)
			df_list.append(df_temp)
			
			label_name = osp.join(self.data_dir,label_prefix + heading + '.csv') # read labels
			label_temp = pd.read_csv(label_name,header=None, names=['label'])
			label_list.append(label_temp)

		df = pd.concat(df_list, ignore_index=True)

		labels = pd.concat(label_list, ignore_index=True)
		df = pd.concat([labels, df], axis="columns") # Append labels to data

		df = df.sample(frac = 1) # shuffle dataframe

		self.df = df

		all_labels = df['label'].to_numpy()
		all_data = df.iloc[:,1:].to_numpy()

		lambdas = all_data[:,3:]

		if rescale: # rescale (normalize) wavelength intensity data
			lam_std = np.std(lambdas, axis=0)  # Calculate standard deviation along columns
			lam_mean = np.mean(lambdas, axis=0)  # Calculate mean along columns
			lambdas_rescaled = (lambdas - lam_mean) / lam_std 
			lambdas = lambdas_rescaled

		# if k_inds != None: # THIS DOES NOT MAKE SENSE HERE
		#     all_data[:,3:] = all_data[:,np.array(k_inds)]
		self.data = lambdas
		self.labels = all_labels

		if include_coords: # removes x,y coordinates (and heading index) from data
			# all_data = all_data[:,3:]
			self.coords = all_data[:,:3]
		else:
			self.coords = None

		if k_inds != None:
			self.data = self.data[:,np.array(k_inds)]

	def __len__(self):
		return len(self.data)

	def __getitem__(self,i):
		if self.inclx:
			state = {'label':self.labels[i],'coords':self.coords[i],'data':self.data[i]}
			return state
		else:
			state = {'label':self.labels[i],'data':self.data[i]}
			return state

	def __getstate__(self):
		if self.inclx:
			state = {
				'data': self.data,
				'label': self.labels,
				'coords':self.coords
					}
			return state
		else:
			state = {
				'data': self.data,
				'label': self.labels,
					}
			return state

	def __setstate__(self, state):
		# Set the object's state from the provided dictionary
		if include_coords:
			self.data = state['data']
			self.labels = state['label']
			self.coords = state['coords']
		else:
			self.data = state['data']
			self.labels = state['label']

class oldPennImageData(Dataset):

    def __init__(self,data_inds = None,rot_augment = False,largeK = False,k_inds = None):
        
        self.data_dir = "./penn_image_data"

        if largeK:
            self.data = torch.load(osp.join(self.data_dir,'img_kvals_largeK.pt'))
            # shape (K,72,72)

            self.labels = torch.load(osp.join(self.data_dir,'img_labels_largeK.pt'))
            # shape (72,72)
        else:
            self.data = torch.load(osp.join(self.data_dir,'img_kvals_interp.pt'))
            # shape (K,72,72)

            self.labels = torch.load(osp.join(self.data_dir,'img_labels_interp.pt'))
            # shape (72,72)

        if rot_augment:

            for i in range(len(self.data)):

                dat = self.data[i]
                labs = self.labels[i]

                for k in [1,2,3]: # number of 90deg rotations

                    self.data.append(torch.rot90(dat,k = k,dims = [1,2]))
                    self.labels.append(torch.rot90(labs,k = k,dims = [0,1]))

        if data_inds is not None:

            self.data = [self.data[i] for i in data_inds]
            self.labels = [self.labels[i] for i in data_inds]

        if k_inds is not None:

            self.data = [x[k_inds,...] for x in self.data]


    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        state = {'label':self.labels[i],'data':self.data[i]}
        return state

    def __getstate__(self):
        state = {
            'data': self.data,
            'label': self.labels,
                }
        return state

    def __setstate__(self, state):
        # Set the object's state from the provided dictionary
        self.data = state['data']
        self.labels = state['label']
# functions to use in presentation notebook

def exampleDataPlot():
	KSPACE = np.arange(0.05,1.05,0.025)
	example_data = oldPennImageData()[15]
	# Define the colors for the three values: pale pink, red, gray
	colors = [(1, 0.7529, 0.7961), (1, 0, 0), (0.5019, 0.5019, 0.5019)]  # RGB for pale pink, red, gray
	n_bins = [3]  # Number of bins
	cmap_name = 'custom_pale_red_gray'
	# Create the colormap
	cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
	fig,ax = plt.subplots(2,4,figsize = (12,6),dpi = 300)
	ax[0,0].imshow(example_data['label'],cmap = cm)
	ax[0,0].axis('off')
	ax[0,0].set_title('Annotated abnormal region')
	top_kinds = [0,5,10]
	bottom_kinds = [15,20,25,30]
	for i,kk in enumerate(top_kinds):
		ax[0,i+1].imshow(example_data['data'][kk],cmap = 'jet')
		ax[0,i+1].axis('off')
		ax[0,i+1].set_title(r"$\lambda = {}$ mm".format(np.round(KSPACE[kk],decimals = 3)))
	for i,kk in enumerate(bottom_kinds):
		ax[1,i].imshow(example_data['data'][kk],cmap = 'jet')
		ax[1,i].axis('off')
		ax[1,i].set_title(r"$\lambda = {}$ mm".format(np.round(KSPACE[kk],decimals = 3)))


def model_threeplot(model_name,metric_filename = 'metrics.h5'):
	fig,ax = plt.subplots(1,3,figsize = (12,3),dpi = 300)
	plt.subplots_adjust(wspace=0.35, hspace=0.5)

	if model_name == "FFNN" or model_name == 'Conv1D':
		if model_name == 'FFNN':
			name = "Feed-forward NN"
			info = dd.io.load('notebook_models/FFNN/metrics.h5')

		elif model_name == 'Conv1D':
			name = "1D Convolutional Network"
			info = dd.io.load('notebook_models/Conv1D/metrics.h5')
		subsets = ['train','val','test']
		ts = info['train_iter']
		ls = info['train_loss']
		tts,avg = info['avg_test_loss']
		ax[0].plot(tts,avg,linewidth = 2,label = 'avg test loss')
		min_time = min([np.min(t) for t in ts])
		max_time = max([np.max(t) for t in ts])
		ts_common = np.linspace(0, 1, 1000)
		interpolated_arrays = []
		cutoff = 10
		for i,(t, l) in enumerate(zip(ts, ls)):
			t = t[cutoff:]/max(t)
			l = l[cutoff:]
			smoothed_l = evl.moving_average(l, 30)
			interp_func = interp1d(t, smoothed_l, bounds_error=False, fill_value=(smoothed_l[0], smoothed_l[-1]))
			interpolated_arrays.append(interp_func(ts_common))

		interpolated_arrays = np.array(interpolated_arrays)
		ls_common = np.nanmean(interpolated_arrays, axis=0)

		ax[0].plot(ts_common[cutoff:],ls_common[cutoff:],linewidth = 2,label = "avg train loss")
		ax[0].set_xlabel("progress fraction")
		ax[0].set_ylabel("Cross-entropy loss")
		ax[0].set_title("Test vs train loss")
		ax[0].legend() # these ones are trained w/ different ecosystem

	else:
		model_metrics = osp.join('notebook_models',model_name,metric_filename)
		info = dd.io.load(model_metrics)
		name = model_name
		subsets = ['train','test']

		ax[0].plot(info['test_loss'],label = 'test')
		ax[0].plot(info['train_loss'],label = 'train')
		ax[0].set_xlabel('training epoch')
		ax[0].set_ylabel("loss")
		ax[0].set_title("Test vs train loss")
		ax[0].legend()
		ax[0].set_yscale('log')

	ts = info['thresholds']
	f1s = info['test_F1']
	f1_thresh_max = ts[np.argmax(f1s)]
	ax[1].plot(ts,f1s,label = 'test')
	ax[1].set_xlabel("classification threshold")
	ax[1].set_ylabel("F1 Score")
	ax[1].set_title("F1 score (max at t = {})".format(np.round(f1_thresh_max,decimals = 3)))
	ax[1].legend()

	
	for subset in subsets:
		tprs = np.append(info[subset+'_tpr'][::-1],1)
		fprs = np.append(info[subset+'_fpr'][::-1],1)
		auc = np.trapz(tprs,fprs)
		ax[2].plot(fprs,tprs,label = subset + "; AUC = {}".format(np.round(auc,decimals = 4)))
	line = np.linspace(0,1,100)
	ax[2].plot(line,line,'k--')
	ax[2].set_xlabel("False positive rate")
	ax[2].set_ylabel("True positive rate")
	ax[2].set_title("ROC Curves; in/out of sample")
	ax[2].legend(loc='upper left', bbox_to_anchor=(1, 0.7))
	fig.suptitle("{} Metrics (5-fold cross-val); {} params".format(name,info['params']),y = 1.1)


def modelImgCompare(name,optimal_threshold): # this is for models which don't take in image data

	if name == "FFNN":
		# LOAD FFNN MODEL
		ff_config = AttrDict({'input_size':39,'extra_layer':True})
		model,model_name = fet.load('notebook_models/FFNN/basic_feedforward.py',ff_config)
		checkpoint_dir = osp.join('notebook_models','FFNN','data_fold1')
		model_ckpt_name = "model_fold1.ckpt-"
		ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(model_ckpt_name) and re.search(r'\d+$', f)]
		latest_ckpt = max(ckpt_files, key=lambda x: int(re.search(r'(\d+)$', x).group()), default=None)
		ckpt_file = osp.join(checkpoint_dir, latest_ckpt)
		model_test_headings = ['aaa0060','aaa0061','aaa0069','aaa_0059']
		test_element = 4
		input_inds = range(39)


	elif name == "Conv1D":
		# LOAD Conv1D MODEL
		conv1d_config = AttrDict({'dim_hidden':32,'n_classes':2,'kernel_size':3})
		model,model_name = fet.load('notebook_models/Conv1D/convNet.py',conv1d_config)
		checkpoint_dir = osp.join('notebook_models','Conv1D','data_fold1')
		model_ckpt_name = "model_fold1.ckpt-"
		ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(model_ckpt_name) and re.search(r'\d+$', f)]
		latest_ckpt = max(ckpt_files, key=lambda x: int(re.search(r'(\d+)$', x).group()), default=None)
		ckpt_file = osp.join(checkpoint_dir, latest_ckpt)
		model_test_headings = ['aaa0060','aaa0061','aaa0069','aaa_0059']
		test_element = 4
		input_inds = range(39)

	# else:

	if name in ['FFNN',"Conv1D"]:
		# LOAD MODEL WEIGHTS
		model_wts = torch.load(ckpt_file)
		model.load_state_dict(model_wts['model_state_dict'])
		model.eval()

				# CONSTRUCT IMAGE DATA FROM TEST HEADING
		data_dir = 'spectra_by_img'
		kval_files = []
		label_files = []
		for heading in model_test_headings:
			for filename in os.listdir(data_dir):  # Loop through all files in the directory
				startstring = "kvals_fuse_rotate_" + heading
				label_startstring = 'labels_fuse_rotate_'+heading
				if filename.startswith(startstring):  # Check if the filename starts with 'file_str'
					# print(filename)
					kval_files.append(filename)
					suffix = filename.replace(startstring,'').replace('.csv','')[:3]
					for fname in os.listdir(data_dir):
						if fname.startswith(label_startstring + suffix):
							label_files.append(fname)
		kval_fname = osp.join(data_dir,kval_files[test_element])
		label_fname = osp.join(data_dir,label_files[test_element])
		df_temp = pd.read_csv(kval_fname,header = None)
		kvals = df_temp.iloc[:,2:]
		df_temp = pd.concat((df_temp.iloc[:,:2],kvals),axis = 'columns')
		labels_df = pd.read_csv(label_fname,header = None,names = ['label'])
		df = pd.concat([labels_df,df_temp],axis = 'columns')
		xmin = df[0].min()
		ymin = df[1].min()
		xshape = df[0].max() - xmin + 1
		yshape = df[1].max() - ymin + 1
		label_arr = 2*np.ones((xshape,yshape))
		KL = len(kvals.iloc[0])
		kval_arr = np.zeros((KL,xshape,yshape))
		for index, row in df.iterrows():
			xx = int(row[0]) - xmin
			yy = int(row[1]) - ymin
			label_arr[xx,yy] = row['label']
			kval_arr[:,xx,yy] = row[3:]

		kvals = torch.tensor(kval_arr)
		labels = torch.tensor(label_arr)
		kvals = F.interpolate(kvals.unsqueeze(0),size = (72,72),mode = 'nearest').squeeze(0)
		labels = F.interpolate(labels.unsqueeze(0).unsqueeze(0),size = (72,72),mode = 'nearest').squeeze(0).squeeze(0)
		kvals = (kvals - kvals.mean().item())/kvals.std().item()

		# CONSTRUCT MODEL PREDICTION IMAGE (FFNN, CONV1D)
		kvals = kvals.permute(1,2,0)
		model_preds = torch.zeros(72,72)
		model_probs = torch.zeros(72,72)
		for i in range(kvals.shape[0]):
			for j in range(kvals.shape[1]):
				if labels[i,j] == 2:
					model_preds[i,j] = 2
					model_probs[i,j] = 2
				else:
					with torch.no_grad():
						probs = F.softmax(model({'data':kvals[i,j].unsqueeze(0)}),dim =1)[0]
					pos_prob = probs[1]
					model_probs[i,j] = pos_prob
					if pos_prob >= optimal_threshold:
						model_preds[i,j] = 1
					else:
						model_preds[i,j] = 0


	# CONSTRUCT PLOTS
	colors = [(1, 0.7529, 0.7961), (1, 0, 0), (0.5019, 0.5019, 0.5019)]  # RGB for pale pink, red, gray
	n_bins = [3]  # Number of bins
	cmap_name = 'custom_pale_red_gray'
	# Create the colormap
	label_cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
	fig,ax = plt.subplots(1,3,figsize = (12,3),dpi = 300)
	plt.subplots_adjust(wspace=0.5, hspace=0.5)

	ax[0].imshow(labels,cmap = label_cm)
	ax[0].axis('off')
	ax[0].set_title("Ground truth")

	# Define the range for the red part
	red_min, red_max = 0, 1

	# Create the original 'Reds' colormap for the gradient red part
	reds = matplotlib.cm.get_cmap('Reds')

	# Create the original 'gray' colormap for the values above 1
	gray = matplotlib.cm.get_cmap('Greys')

	# Combine them: first take the whole 'Reds' colormap, then append the 'gray' color
	# The red values will correspond to normalized values between 0 and 1
	# The gray values will be used for normalized values above 1
	combined_colors = np.vstack((reds(np.linspace(0, 1, 256)), gray(np.ones(256))))  # Use gray(np.ones(256)) for a consistent gray color
	custom_cmap = LinearSegmentedColormap.from_list('reds_gray', combined_colors)

	ax[1].imshow(model_probs,cmap = custom_cmap)
	ax[1].axis('off')
	ax[1].set_title("Model probability heatmap")

	ax[2].imshow(model_preds,cmap = label_cm)
	ax[2].axis('off')
	ax[2].set_title("Model preds at t = {}".format(np.round(optimal_threshold,decimals = 3)))


def ImageModelCompare(model_config,data_config,config,experiment_dir,optimal_threshold,test_elem = 1):
	model,model_name = fet.load(model_config,config) # load model
	# if mps:
	# 	model = model.to('mps')
	fold = 0
	wts_file = osp.join(experiment_dir,model_name + '_wts_fold{}.pth'.format(fold+1))
	model_wts = torch.load(wts_file)
	model.load_state_dict(model_wts)

	kfold_loaders,k_inds = fet.load(data_config, config) # load data
	example_loader = kfold_loaders[fold]['test']
	example_data,example_labels = next(iter(example_loader))
	# test_elem = 1
	test_data = example_data[test_elem]
	labels = example_labels[test_elem]

	with torch.no_grad():
		preds = F.softmax(model(example_data),dim = 1)[test_elem]

	model_probs = preds[1,...]
	model_preds = torch.zeros(model_probs.shape)
	thresh_mask = (model_probs > optimal_threshold)
	model_preds[thresh_mask] = 1

	# CONSTRUCT PLOTS
	colors = [(1, 0.7529, 0.7961), (1, 0, 0)]  # RGB for pale pink, red, gray
	n_bins = [2]  # Number of bins
	cmap_name = 'custom_pale_red_gray'
	# Create the colormap
	label_cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
	fig,ax = plt.subplots(1,3,figsize = (12,3),dpi = 300)
	plt.subplots_adjust(wspace=0.5, hspace=0.5)

	ax[0].imshow(labels,cmap = label_cm)
	ax[0].axis('off')
	ax[0].set_title("Ground truth")

	# # Define the range for the red part
	# red_min, red_max = 0, 1

	# # Create the original 'Reds' colormap for the gradient red part
	# reds = matplotlib.cm.get_cmap('Reds')

	# # Create the original 'gray' colormap for the values above 1
	# gray = matplotlib.cm.get_cmap('Greys')

	# # Combine them: first take the whole 'Reds' colormap, then append the 'gray' color
	# # The red values will correspond to normalized values between 0 and 1
	# # The gray values will be used for normalized values above 1
	# combined_colors = np.vstack((reds(np.linspace(0, 1, 256)), gray(np.ones(256))))  # Use gray(np.ones(256)) for a consistent gray color
	# custom_cmap = LinearSegmentedColormap.from_list('reds_gray', combined_colors)

	ax[1].imshow(model_probs,cmap = 'jet')
	ax[1].axis('off')
	ax[1].set_title("Model probability heatmap")

	ax[2].imshow(model_preds,cmap = label_cm)
	ax[2].axis('off')
	ax[2].set_title("Model preds at t = {}".format(np.round(optimal_threshold,decimals = 3)))



