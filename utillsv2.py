
from matplotlib.pyplot import axis
import numpy as np

import torch.utils.data as data
import pandas as pd
import option
from scipy.stats import multivariate_normal


import os

import matplotlib.pyplot as plt
import torch


args = option.parser.parse_args()



def Concat_list_all_crop_feedback(Test=False, create='False'): #UCF
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    if Test is True:
        con_test = np.load("concatenated/Concat_test_10.npy")
        # con_test = np.load("/l/users/anas.al-lahham/concat_test_XD_5crop.npy")
        print('Testset size:', con_test.shape)
        # con_test
        return con_test
    if Test is False:

        if create == 'True':
            print('loading Pseudo Labels......',args.pseudofile )
        label_all = np.load(args.pseudofile)

        print('[*] concatenated labels shape:',label_all.shape)

        return len(label_all), torch.tensor(label_all).cuda()

