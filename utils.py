
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



# con_arr having all of them 
def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_AUC"][-1]))
    fo.close()









def load_labels():
    
    label_normal= np.zeros((580699,))
    label_abnormal = np.load(args.pseudo)
    label_all = np.concatenate((label_normal,label_abnormal), axis = 0)
    print('[*] label data shape:', label_all.shape)
    return label_all





def Concat_list_all_crop_feedback(Test=False, create='False'): #UCF
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    if Test is True:
        # con_test = np.load("/home/anas.al-lahham/AnomalyDetection/RTFM/10_crop_features/Concat_test_10.npy")
        con_test = np.load("/l/users/anas.al-lahham/concat_test_XD_5crop.npy") #C3D XD
        # con_test = con_test[:, 0, :]
        # con_test = np.load("/home/anas.al-lahham/Baseline_AD/RFS_AD/extract_C3Dv2/extruct-video-feature/C3D_concatenated/UCF_C3D_Test1.npy") #C3D UCF

        # mean = np.mean(con_test)
        # std = np.std(con_test)
        # con_test = (con_test - mean) / std
        # print(con_test.shape)
        return con_test
    if Test is False:
        # n_train_crop = np.load("/home/anas.al-lahham/AnomalyDetection/RTFM/10_crop_features/Concat_N_10.npy")
        # print('[*] normal data shape:', n_train_crop.shape)
        
        # a_train_crop = np.load("/home/anas.al-lahham/AnomalyDetection/RTFM/10_crop_features/Concat_A_10.npy")
        # print('[*] Abnormal data shape:', a_train_crop.shape)
        if create == 'True':
            print('loading Pseudo Labels......')
        label_all = np.load(args.pseudofile)

        print('[*] concatenated labels shape:',label_all.shape)

        return len(label_all), torch.tensor(label_all).cuda()




# def Concat_list_all_crop_feedback(Test=False, create='False'): #XD
#     from datetime import datetime

#     now = datetime.now()

#     current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
#     if Test is True:
#         con_test = np.load("/l/users/anas.al-lahham/concat_test_XD_5crop.npy")
#         print(con_test.shape)
#         return con_test
#     if Test is False:
#         n_train_crop = np.load("/l/users/anas.al-lahham/concat_N_XD_5crop.npy")
#         print('[*] normal data shape:', n_train_crop.shape)
    
#         a_train_crop = np.load("/l/users/anas.al-lahham/concat_A_XD_5crop.npy")
#         print('[*] Abnormal data shape:', a_train_crop.shape)
#         if create == 'True':
#             print('Creating Pseudo Labels......')

#             n_train_crop_features = get_matrix(n_train_crop)
#             print('[*] normal data features:',n_train_crop_features.shape)

#             mu, var = estimate_gauss(n_train_crop_features)
#             print('[*] Mean:',mu,'[*] Var:', var) 

#             p = multivariate_normal(mu, var)
#             n_probs = p.pdf(n_train_crop_features)
#             print('[*] n_probs shape:', n_probs.shape)
            
#             # eps_N = args.eps
        

#             # n_train_crop_features = get_matrix(n_train_crop[np.where((n_probs > eps_N))[0]])

#             # mu, var = estimate_gauss(n_train_crop_features)
#             # print('[*] new mu , var:', mu, var)  # np.sqrt(var)
#             # p = multivariate_normal(mu, var)


#             idx_list = np.load('XD_IDX.npy')




#             def pseudo_labelling_new(idx_list,a_train,window_size_s):
#                 ground_truth = []

#                 for video in idx_list:
#                     sample = a_train[video[0]:video[1]]

#                     # feature extraction 
#                     sample_matrix = get_matrix(sample)  # for just l2
#                     # sample_matrix = get_matrix(sample)  # along with l1

#                     # get p values
#                     probs = p.pdf(sample_matrix)


#                     window_size_f = window_size_s
#                     temp_list = []
#                     temp_list += [0.0] * len(probs)
                    

#                     window_size = int(len(probs) * window_size_f)  # fixed
#                     temp = []
#                     for idx in range(0, len(probs) - window_size + 1):
#                         arr = 0
#                         for i in range(idx, idx + window_size - 1):
#                             arr += abs(probs[i+1] - probs[i])
#                         temp.append(arr)

#                     for i in range(temp.index(max(temp)), temp.index(max(temp)) + window_size):
#                         temp_list[i] = 1.0

#                     ground_truth += temp_list




#                 print('[*] Length of Pseudo labels:', len(ground_truth))
#                 return ground_truth
            

#             window_size = args.windowsize
#             print('[*] Window Size:', window_size)


#             abnormal_pseudo_labels = np.array(pseudo_labelling_new(idx_list,a_train_crop,window_size))
            
        
#             con_label_A = abnormal_pseudo_labels
            
            
#             # con_label_A[nomal_in_conA]= 0.0
#             # np.save('/home/anas.al-lahham/Baseline_AD/RFS_AD/Pseudo_labels/'+'Pseudo_Lebels'+str(current_time)+'_crop.npy', abnormal_pseudo_labels)
#             # args.pseudofile ='/home/anas.al-lahham/Baseline_AD/RFS_AD/iterative_UCF_labels/'+'{}.npy'.format(args.pseudofile)

#             np.save(args.pseudofile, con_label_A)
#         else:

#             print('Loading Pseudo Labels......')
#             print(args.pseudofile)
#             con_label_A = np.load(args.pseudofile)

#         # nomal_in_conA_features = np.load('/home/anas.al-lahham/Baseline_AD/RFS_AD/Normal_in_Anomaly.npy') #Top x% of normal data in anomaly data
#         # n_train_crop = np.concatenate((n_train_crop, a_train_crop[nomal_in_conA_features]), axis=0) # add top x% of normal data in anomaly data to normal data
        

#         # con_label_A[nomal_in_conA_features]= 0.0
#         # a_train_crop = np.delete(a_train_crop, nomal_in_conA_features, axis= 0)
#         # con_label_A = np.delete(con_label_A, nomal_in_conA_features)

#         con_label_N = np.zeros((len(n_train_crop),))
#         # print('[*] n_train_crop  shape:', n_train_crop.shape)
#         # print('[*] n_train_crop pseudo labels  shape:', con_label_N.shape)
        
#         # print('[*] a_train_crop  shape:', a_train_crop.shape)
#         # print('[*] a_train_crop pseudo labels shape:', con_label_A.shape)
#         # con_all = np.concatenate((n_train_crop,a_train_crop), axis = 0)
#         label_all = np.concatenate((con_label_N,con_label_A), axis = 0)
#         # print('[*] concatenated dataset shape:', con_all.shape)
#         print('[*] concatenated labels shape:',label_all.shape)
#         n_train_crop = []
#         a_train_crop = []
#         # np.save('iterative_UCF_labels/'+'{}.npy'.format(args.conall), con_all)
#         # con_all = []
#         # np.save('iterative_UCF_labels/lebel_all.npy', label_all)
#         return len(con_label_N), torch.tensor(label_all).cuda()






def Concat_list_all_crop_XD(Test=False, create='False'):
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    if Test is True:
        con_test = np.load("/l/users/anas.al-lahham/concat_test_XD_5crop.npy")
        print(con_test.shape)
        return con_test
    if Test is False:
        n_train_crop = np.load("/l/users/anas.al-lahham/concat_N_XD_5crop.npy")
        print('[*] normal data shape:', n_train_crop.shape)
        
        a_train_crop = np.load("/l/users/anas.al-lahham/concat_A_XD_5crop.npy")
        print('[*] Abnormal data shape:', a_train_crop.shape)
        if create == 'True':
            print('Creating Pseudo Labels......')

            n_train_crop_features = get_matrix(n_train_crop)
            print('[*] normal data features:',n_train_crop_features.shape)

            mu, var = estimate_gauss(n_train_crop_features)
            print('[*] Mean:',mu,'[*] Var:', var) 

            p = multivariate_normal(mu, var)
            n_probs = p.pdf(n_train_crop_features)
            print('[*] n_probs shape:', n_probs.shape)
            
        

            idx_list = np.load('10crop_Abnormal_video_idx.npy')


            a_train_crop_features = get_matrix(a_train_crop)

            a_probs = p.pdf(a_train_crop_features)



            eps_A = args.eps2

            b_A= (len(a_train_crop_features),)
            a_A= a_probs[a_probs > eps_A].shape
            print('[*] Percentage of a_probs:',a_A[0]/b_A[0],'%')


            nomal_in_conA = np.where((a_probs > eps_A))[0]

            nomal_in_conA_features = a_train_crop[nomal_in_conA]


            n_train_crop = np.concatenate((n_train_crop, nomal_in_conA_features), axis=0)

            print('[*] new normal dataset shape:', n_train_crop.shape)


            new_n_train_crop_features = get_matrix(n_train_crop)

            print('[*] new normal data features:',new_n_train_crop_features.shape)
            mu1, var1 = estimate_gauss(new_n_train_crop_features)
            print(mu1, var1)  # np.sqrt(var)
            p = multivariate_normal(mu1, var1)

            n_probs = p.pdf(new_n_train_crop_features)
           




            ##### 10% of n_probs ######
            eps_N = args.eps
            # print('[*] eps of new n_probs:',eps_N)
            # b_N= (len(new_n_train_crop_features),)
            # a_N= n_probs[n_probs < eps_N].shape
            # print('[*] Percentage of new n_probs:',a_N[0]/b_N[0],'%')


            def pseudo_labelling_new(idx_list,a_train,window_size_s):
                ground_truth = []

                for video in idx_list:
                    sample = a_train[video[0]:video[1]]

                    # feature extraction 
                    sample_matrix = get_matrix(sample)  # for just l2
                    # sample_matrix = get_matrix(sample)  # along with l1

                    # get p values
                    probs = p.pdf(sample_matrix)


                    # b_sample= (len(probs),)
                    # a_sample= probs[probs > eps_A].shape
                    # window_size_sample = a_sample[0]/b_sample[0]
                    # if  window_size_sample <= 0.1 or window_size_sample >= 0.2:
                    #     window_size_f = window_size_s

                    # else: 
                    #     window_size_f = window_size_sample
                    # print('[*] Windowsize:',window_size_f,'%')
                    window_size_f = window_size_s
                    temp_list = []
                    temp_list += [0.0] * len(probs)
                    

                    window_size = int(len(probs) * window_size_f)  # fixed
                    temp = []
                    for idx in range(0, len(probs) - window_size + 1):
                        arr = 0
                        for i in range(idx, idx + window_size - 1):
                            arr += abs(probs[i+1] - probs[i])
                        temp.append(arr)

                    for i in range(temp.index(max(temp)), temp.index(max(temp)) + window_size):
                        temp_list[i] = 1.0

                    ground_truth += temp_list
                    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                    axs[0].set_title(str(video)+'labels')
                    axs[0].plot(temp_list, 'r')
                    axs[1].set_title(str(video)+'probs')
                    axs[1].plot(probs, 'b')
                    plt.savefig(os.path.join('/home/anas.al-lahham/Baseline_AD/RFS_AD/video_probs', str(video)))
                    plt.close(fig)



                axs[0].set_title(str(video))
                print('[*] Length of Pseudo labels:', len(ground_truth))
                return ground_truth
            

            window_size = args.windowsize
            print('[*] Window Size:', window_size)

            # abnormal_pseudo_labels = np.array(pseudo_labelling_new_random(idx_list,a_train_crop,window_size_r))
            abnormal_pseudo_labels = np.array(pseudo_labelling_new(idx_list,a_train_crop,window_size))
            
        
            con_label_A = abnormal_pseudo_labels
            
            
            con_label_A[nomal_in_conA]= 1.0
            # np.save('/home/anas.al-lahham/Baseline_AD/RFS_AD/Pseudo_labels/'+'Pseudo_Lebels'+str(current_time)+'_crop.npy', abnormal_pseudo_labels)
            np.save('/home/anas.al-lahham/Baseline_AD/RFS_AD/Pseudo_labels/'+'Pseudo_Lebels_'+'Windowsize:'+str(args.windowsize)+'eps_A'+str(args.eps2)+'_crop.npy', abnormal_pseudo_labels)
        else:

            print('Loading Pseudo Labels......')
            print(args.pseudo)
            con_label_A = np.load(args.pseudo)


            
        # a_train_crop = np.delete(a_train_crop, np.where((a_probs > eps_A))[0], axis= 0)
        # con_label_A = np.delete(con_label_A, np.where((a_probs > eps_A))[0])

        con_label_N = np.zeros((len(n_train_crop),))
        # con_label_A[np.where((con_label_A == 1.0))[0]] = 0.0
        # con_label_A[np.where((con_label_A == 0.0))[0]] = 1.0
        

        
        print('[*] a_train_crop  shape:', a_train_crop.shape)
        print('[*] a_train_crop pseudo labels  shape:', con_label_A.shape)
        con_all = np.concatenate((n_train_crop,a_train_crop), axis = 0)
        label_all = np.concatenate((con_label_N,con_label_A), axis = 0)
        print('[*] concatenated dataset shape:', con_all.shape)
        print('[*] concatenated labels shape:',label_all.shape)
        n_train_crop = []
        a_train_crop = []

        return con_all, label_all





def window_fitter(confidence_scores, ):
    print(args.pseudofile)
    gt = np.load(args.pseudofile)  # gaussian-labeled abnormal features
    abnormal_list = list(open('/l/users/anas.al-lahham/new_RGB.txt'))[:1905] # list of abnormal videos [just to get the video lengths]
    
    # to control ground truth's indices
    from_id = 0
    to_id = 0
    
    fc_gt = []  # new pseudo-labels taken from fc network
    iou_scores = []
    for video in abnormal_list:
        location = video.strip('\n')
        sample = np.load(location)
        
        num_features = len(sample)
        temp_list = []
        temp_list += [0.0] * num_features
        
        to_id += num_features
        
        # get specific video separately
        probs = confidence_scores[from_id:to_id]
        
        # to make it fixed/variable
        variable_window_size = np.count_nonzero(gt[from_id:to_id])
        window_size =    int(num_features * 0.1)  # fixed
        temp = []
        for idx in range(0, num_features - window_size + 1):
            arr = 0
            for i in range(idx, idx + window_size - 1):
                arr += abs(probs[i+1] - probs[i])
            temp.append(arr)
        
        for i in range(temp.index(max(temp)), temp.index(max(temp)) + window_size):
            temp_list[i] = 1.0
        
        # to get stats 
        # np.where(res > 0.1, 1.0, 0.0)
        prev_lls = gt[from_id:to_id]
        next_lls = np.array(temp_list)
        res = np.where(prev_lls > 0.1, 1.0, 0.0) + next_lls
        if np.count_nonzero(res) > 0:
            iou = res[res == 2.].shape[0] / (res[res == 1.].shape[0] + res[res == 2.].shape[0])
            iou_scores.append(iou)
            with open('iou_scores.txt', 'a') as f:
                f.write(f"{iou}\n")
        else:
            print('no anomaly')
        
        new_rep = prev_lls + next_lls
        if iou < 0.05:
            new_rep[new_rep > 0.0] = 1.0
        else:
            new_rep /= 2    
            # new_rep[(new_rep > 0.5) & (new_rep < 1.0)] = 0.8


        # res /= 2
        # res[res == 0.5] = 0.75
        fc_gt += list(new_rep)
        
        # fc_gt += list(np.where(res > 0.1, 1.0, 0.0))
        from_id += num_features
    
    return np.array(fc_gt), sum(iou_scores) / len(iou_scores)




def gaussian_window_fitter(confidence_scores, ):
# get normal pdf 
    n_train = np.load("/home/anas.al-lahham/AnomalyDetection/RTFM/10_crop_features/Concat_N_10.npy")
    n_l2norms = np.sum(np.square(n_train), axis=2)
    n_l2norms = np.mean(n_l2norms, axis= 1)
    mu, var = estimate_gauss(n_l2norms)
    # probability model
    p = multivariate_normal(mu, var)
    
    
    gt = np.load('/home/anas.al-lahham/Baseline_AD/RFS_AD/iterative_UCF_labels/'+'{}.npy'.format(args.pseudofile))  # gaussian-labeled abnormal features

    if args.feature_size == 2048:
        abnormal_list = list(open('list/ucf-i3d.list'))[:810] 
        # list of abnormal videos [just to get the video lengths]
    
        # to control ground truth's indices
        from_id = 0
        to_id = 0
        
        fc_gt = []  # new pseudo-labels taken from fc network
        iou_scores = []
        for video in abnormal_list:
            location = video.strip('\n')
            sample = np.load(location)
            sample_matrix = np.sum(np.square(sample), axis=2) 
            sample_matrix = np.mean(sample_matrix, axis= 1) # for just l2
            
            num_features = len(sample)
            temp_list = []
            temp_list += [0.0] * num_features
            to_id += num_features
            
            # get specific video separately
            probs = confidence_scores[from_id:to_id]
            sample_matrix = sample_matrix * probs
            
            # get p values
            probs = p.pdf(sample_matrix)
            
            # to make it fixed/variable
            variable_window_size = np.count_nonzero(gt[from_id:to_id])
            window_size = int(num_features * 0.1)  # fixed
            temp = []
            for idx in range(0, num_features - window_size + 1):
                arr = 0
                for i in range(idx, idx + window_size - 1):
                    arr += abs(probs[i+1] - probs[i])
                temp.append(arr)
            
            for i in range(temp.index(max(temp)), temp.index(max(temp)) + window_size):
                temp_list[i] = 1.0
            
            # to get stats 
            # np.where(res > 0.1, 1.0, 0.0)
            prev_lls = gt[from_id:to_id]
            next_lls = np.array(temp_list)
            res = np.where(prev_lls > 0.1, 1.0, 0.0) + next_lls
            if np.count_nonzero(res) > 0:
                iou = res[res == 2.].shape[0] / (res[res == 1.].shape[0] + res[res == 2.].shape[0])
                iou_scores.append(iou)
                with open('iou_scores.txt', 'a') as f:
                    f.write(f"{iou}\n")
            else:
                print('no anomaly')
            
            new_repr = prev_lls + next_lls
            if iou == 0.0:
                new_repr[new_repr > 0.0] = 1.0
            else:
                new_repr /= 2 
                # new_repr[(new_repr > 0.5) & (new_repr < 1.0)] = 0.8
                new_repr[new_repr == 0.5] = 0.8

            new_repr[new_repr < 0.5] = 0.0
            

            fc_gt += list(new_repr)
            
            # fc_gt += list(np.where(res > 0.1, 1.0, 0.0))
            from_id += num_features
    
        return np.array(fc_gt), sum(iou_scores) / len(iou_scores)



def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_auc"][-1]))
    # fo.write(str(test_info["test_PR"][-1]))
    fo.close()