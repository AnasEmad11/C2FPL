import torch.utils.data as data
import numpy as np
from utillsv2 import  Concat_list_all_crop_feedback
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from tqdm import tqdm
import option

args = option.parser.parse_args()


class Dataset_Con_all_feedback_XD(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal


        if test_mode:

            self.con_all = Concat_list_all_crop_feedback(True)
        else:
            

            
            self.con_all = np.load("concatenated/concat_UCF.npy")
            # self.con_all = np.load("concatenated/concat_XD.npy")

            print('self.con_all shape:',self.con_all.shape)

        self.tranform = transform
        self.test_mode = test_mode
        

    def __getitem__(self, index):
        
        
        if self.test_mode:
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)

        else:
            
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)
    



        if self.test_mode:
            
            return features
        else:

            return features , index



    def __len__(self):
        return len(self.con_all)   
