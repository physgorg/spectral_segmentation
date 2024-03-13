import os
import torch
import pandas as pd
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import RepeatedKFold

from os import path as osp
import forge
from forge import flags


# aspects of data managmement
flags.DEFINE_boolean("include_coords",False, "If False, remove coordinate information")
flags.DEFINE_boolean("rescale", True, "Rescales intensities")
flags.DEFINE_boolean("cross_validate",True,"Do K-fold cross-validation")
# flags.DEFINE_boolean("patient_batch",True,"Batch data by patient (data heading)")

# Job management
# flags.DEFINE_integer("batch size",5,"training batch size")
flags.DEFINE_float("train_split",0.8,'Fraction of data to use for training.')
flags.DEFINE_integer("n_repeats",1,"Times to repeat K-fold cross-validation")
flags.DEFINE_integer("split_seed",1,"Seed for KFold split (integer for reproducibility")

# "wavelengths between 0.05 and 1 mm in 0.025 mm increments."
LSPACE = np.arange(0.05,1.05,0.025) # corresponds to 39 non-coord columns of data
L_INDS = None#None # specific indices of data to select

DATAFOLDER = './penn_data'

HEADINGS = np.array(['aaa0043','aaa0044','aaa0060','aaa0061','aaa0063','aaa0064','aaa0066','aaa0069','aaa0086','aaa0087','aaa_0051','aaa_0053','aaa_0054','aaa_0059','aaa_0071','aaa_0072'])

def pad_dataframe(df): # make all data the same shape, add label "2" for no data
    # Find the max for both dimensions
    max_dim = max(df[0].max(), df[1].max())

    # Create all possible combinations of x and y within the new bounds
    mesh_x, mesh_y = np.meshgrid(range(max_dim + 1), range(max_dim + 1))  # New square grid
    all_combinations = pd.DataFrame({
        0: mesh_x.ravel(),
        1: mesh_y.ravel()
    })

    # Merge with the original dataframe to find missing combinations
    merged_df = pd.merge(all_combinations, df, how='left', on=[0, 1])

    # Fill missing 'label' values with 2 (denoting 'no data')
    merged_df['label'] = merged_df['label'].fillna(2)

    # Identify columns other than 0, 1, and 'label' to fill with zeros
    fill_zero_columns = [col for col in df.columns if col not in [0, 1, 'label']]

    # Fill missing values for these columns with 0
    merged_df[fill_zero_columns] = merged_df[fill_zero_columns].fillna(0)

    # Fix any potential column naming due to merge
    if any(isinstance(col, int) for col in merged_df.columns):  # Check if any column is integer
        merged_df.columns = [str(col) if isinstance(col, int) else col for col in merged_df.columns]

    # Remove any additional columns created from merging
    drop_columns = [col for col in merged_df if col.endswith('_drop')]
    merged_df.drop(columns=drop_columns, inplace=True)

    return merged_df,max_dim

class PennData(Dataset):

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


class headingSampler(Sampler): # sample data so batches each come from unique patient
    def __init__(self, data_source):
        super().__init__()
        self.data_source = data_source
        # Sorting data so it is batched by patient
        self.sorted_indices = sorted(range(len(data_source)), key=lambda idx: data_source[idx]['coords'][0])

    def __iter__(self):
        # Yielding indices batch-wise, but indices within each batch are sorted based on h_idx
        for idx in self.sorted_indices:
            yield idx

    def __len__(self):
        return len(self.data_source)

def load(config):

	n_splits = int(len(HEADINGS)/((1-config.train_split)*len(HEADINGS)))

	kf = RepeatedKFold(n_splits = n_splits, n_repeats = config.n_repeats, random_state = config.split_seed) # K-fold cross validation

	kf_dataloaders = []

	for i, (train_ind, test_ind) in enumerate(kf.split(HEADINGS)): # for each K-fold split, 
		n_val = len(test_ind)//2
		n_test = len(test_ind)-n_val   
		n_train = len(train_ind)


		sampler = None
		
		# set up training DataLoader
		train_data = PennData(DATAFOLDER,HEADINGS[train_ind],k_inds = L_INDS,include_coords = config.include_coords,rescale = config.rescale)
		if config.include_coords:
			sampler = headingSampler(train_data)
		train_loader = DataLoader(train_data,batch_size = min(config.batch_size, len(train_data)),num_workers=0,sampler = sampler)

		# set up test DataLoader
		test_data = PennData(DATAFOLDER,HEADINGS[test_ind[:n_test]],k_inds = L_INDS,include_coords = config.include_coords,rescale = config.rescale)
		if config.include_coords:
			sampler = headingSampler(test_data)
		test_loader = DataLoader(test_data,batch_size = min(config.batch_size, len(test_data)),num_workers=0,sampler = sampler)

		# set up val DataLoader
		val_data = PennData(DATAFOLDER,HEADINGS[test_ind[n_test:]],k_inds = L_INDS,include_coords = config.include_coords,rescale = config.rescale)
		if config.include_coords:
			sampler = headingSampler(val_data)
		val_loader = DataLoader(val_data,batch_size = min(config.batch_size, len(val_data)),num_workers=0,sampler = sampler)
		

		dataloaders = {'train':train_loader, 
					   'test':test_loader,
					   'val':val_loader,
					   'train_inds':train_ind,
					   'test_inds':test_ind,
					   'n_val':n_val,
					   'test_headings':list(HEADINGS[test_ind])
					  }
		kf_dataloaders.append(dataloaders)

		if not config.cross_validate:
			break

	return kf_dataloaders,L_INDS
