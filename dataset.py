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
        # if test_mode:
        #     self.rgb_list_file = args.test_rgb_list

        if test_mode:

            self.con_all = Concat_list_all_crop_feedback(True)
        else:
            
            # self.con_all = con_all
            # self.con_all = np.load('{}.npy'.format(args.conall))
            
            self.con_all = np.load("/home/anas.al-lahham/Baseline_AD/RFS_AD/iterative_UCF_labels/concat_UCF.npy")#[:,0,:1024]
            # self.con_all = np.load("/home/anas.al-lahham/AD_Unsupervised/concat_XD.npy")
            # mean = np.mean(self.con_all)
            # std = np.std(self.con_all)
            # self.con_all = (self.con_all - mean) / std
            print('self.con_all shape:',self.con_all.shape)

        self.tranform = transform
        self.test_mode = test_mode
        

    def __getitem__(self, index):
        
        
        if self.test_mode:
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)
        # if self.test_mode:
        #     features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        #     features = np.array(features, dtype=np.float32)
        else:
            
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)
    


        # print('features shape:', features.shape, labels)
        if self.test_mode:
            
            return features
        else:

            return features , index



    def __len__(self):
        return len(self.con_all)   
