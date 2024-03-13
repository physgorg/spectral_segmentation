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
flags.DEFINE_boolean("cross_validate",True,"Do K-fold cross-validation")
flags.DEFINE_boolean("rot_augment",True,"Augment data by random 90 deg rotations.")


# Job management
flags.DEFINE_float("train_split",0.8,'Fraction of data to use for training.')
flags.DEFINE_integer("n_repeats",1,"Times to repeat K-fold cross-validation")
flags.DEFINE_integer("split_seed",1,"Seed for KFold split (integer for reproducibility")

# "wavelengths between 0.05 and 1 mm in 0.025 mm increments."
LSPACE = np.arange(0.05,1.05,0.025) # corresponds to 39 non-coord columns of data
L_INDS = None #None # specific indices of data to select

HEADINGS = np.array(['aaa0043','aaa0044','aaa0060','aaa0061','aaa0063','aaa0064','aaa0066','aaa0069','aaa0086','aaa0087','aaa_0051','aaa_0053','aaa_0054','aaa_0059','aaa_0071','aaa_0072'])


class PennImageData(Dataset):

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

class PennImageDataU(Dataset):

    def __init__(self,data_inds = None,rot_augment = False,largeK = False,k_inds = None):
        
        self.data_dir = "./penn_image_data"

        if largeK:
            self.data = torch.load(osp.join(self.data_dir,'img_kvals_interp_largeK.pt'))
            # shape (K,72,72)

            self.labels = torch.load(osp.join(self.data_dir,'img_labels_interp_largeK.pt'))
            # shape (72,72)
        else:
            self.data = torch.load(osp.join(self.data_dir,'img_kvals_interp_avgK.pt'))
            # shape (K,72,72)

            self.labels = torch.load(osp.join(self.data_dir,'img_labels_interp_avgK.pt'))
            # shape (72,72)

        if rot_augment:

            for i in range(len(self.data)):

                dat = self.data[i]
                labs = self.labels[i]

                for k in [1,2,3]: # number of 90deg rotations

                    self.data.append(torch.rot90(dat,k = k,dims = [1,2]))
                    self.labels.append(torch.rot90(labs,k = k,dims = [0,1]))

        if data_inds is not None:

            self.data = [self.data[i].float() for i in data_inds]
            self.labels = [self.labels[i].long() for i in data_inds]

        if k_inds is not None:

            self.data = [x[k_inds,...] for x in self.data]

        


    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        state = {'label':self.labels[i],'data':self.data[i]}
        return self.data[i],self.labels[i]

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

def load(config):

    allDat = PennImageDataU(rot_augment = config.rot_augment)

    n_images = len(allDat)

    n_splits = int(1/((1-config.train_split)))

    K_INDS = range(30) # wavelength measurements to consider

    kf = RepeatedKFold(n_splits = n_splits, n_repeats = config.n_repeats, random_state = config.split_seed) # K-fold cross validation

    kf_dataloaders = []

    for i, (train_inds, test_inds) in enumerate(kf.split(range(n_images))): # for each K-fold split, 
        
        n_test = len(test_inds)
        n_train = len(train_inds)

        
        # set up training DataLoader
        train_data = PennImageDataU(data_inds = train_inds, k_inds = K_INDS,largeK = False,rot_augment = True)
        train_loader = DataLoader(train_data,batch_size = min(config.batch_size, len(train_data)),num_workers=0)

        # set up test DataLoader
        test_data = PennImageDataU(data_inds = test_inds, k_inds = K_INDS,largeK = False,rot_augment = True)
        test_loader = DataLoader(test_data,batch_size = min(config.batch_size, len(test_data)),num_workers=0)
        
        # # set up val DataLoader
        # val_data = PennImageData(data_inds = test_inds[:n_val])
        # val_loader = DataLoader(val_data,batch_size = min(config.batch_size, len(val_data)),num_workers=0,shuffle = True)

        dataloaders = {'train':train_loader, 
                       'test':test_loader,
                       'train_inds':train_inds,
                       'test_inds':test_inds
                      }
        kf_dataloaders.append(dataloaders)

        if not config.cross_validate:
            break

    return kf_dataloaders,K_INDS




# def createData():
    # # get paired filenames
    # kval_files = []
    # label_files = []
    # for heading in headings:
    #     for filename in os.listdir(data_dir):  # Loop through all files in the directory
    #         startstring = "kvals_fuse_rotate_" + heading
    #         label_startstring = 'labels_fuse_rotate_'+heading
    #         if filename.startswith(startstring):  # Check if the filename starts with 'file_str'
    #             # print(filename)
    #             kval_files.append(filename)
    #             suffix = filename.replace(startstring,'').replace('.csv','')[:3]
    #             for fname in os.listdir(data_dir):
    #                 if fname.startswith(label_startstring + suffix):
    #                     label_files.append(fname)
    # # list(zip(kval_files,label_files))
    # data_tensors = []
    # label_tensors = []
    # for i in tqdm(range(len(kval_files))):
        
    #     kval_fname = osp.join(data_dir,kval_files[i])
    #     label_fname = osp.join(data_dir,label_files[i])

    #     df_temp = pd.read_csv(kval_fname,header = None)

    #     kvals = df_temp.iloc[:,2:]

    #     kvals = (kvals - kvals.mean())/kvals.std()

    #     df_temp = pd.concat((df_temp.iloc[:,:2],kvals),axis = 'columns')

    #     labels_df = pd.read_csv(label_fname,header = None,names = ['label'])

    #     df = pd.concat([labels_df,df_temp],axis = 'columns')
        
    #     df[[0, 1]] = df[[0, 1]] + 2 # create initial padding

    #     label_arr = 2*np.ones((72,72))
    #     kval_arr = np.zeros((72,72,len(kvals.iloc[0])))
    #     for index, row in df.iterrows():
    #         xx = int(row[0])
    #         yy = int(row[1])
    #         label_arr[xx,yy] = row['label']
    #         kval_arr[xx,yy,:] = row[3:]

    #     data_tensors.append(torch.tensor(kval_arr))
    #     label_tensors.append(torch.tensor(label_arr))  
