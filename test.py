import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from dataset import  Dataset_Con_all_feedback_XD
from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import time
import os
from model import Model, Model_V2
# from datasets.dataset import 

def test(dataloader, model, args, device):
    with torch.no_grad():#  #
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            # print(input.size())
            startime = time.time()
            logits = model(inputs=input)
            # print("done in {0}.".format(time.time() - startime))
            pred = torch.cat((pred, logits))


            
        gt = np.load(args.gt)
        # print(gt.shape)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        # gt = gt[:len(pred)] 

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc: ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)

        # np.save('UCF_pred/'+'{}-pred_UCFV1_i3d.npy'.format(epoch), pred)
        return rec_auc, pr_auc
    


def test_2(dataloader, model, args, device):
    with torch.no_grad():  
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)

            logits = model(inputs=input)

            pred = torch.cat((pred, logits))


            
        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        # gt = gt[:len(pred)] 

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        
        np.save('fpr1.npy', fpr)
        np.save('tpr1.npy', tpr)
        rec_auc = auc(fpr, tpr)
        # print('auc: ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print("AP = ",pr_auc)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        np.save('pred_XD.npy', pred)
        
        return rec_auc, pr_auc
    

if __name__ == '__main__':
    args = option.parser.parse_args()
    gt = np.load(args.gt)
    # con_all = np.load('{}.npy'.format(args.conall))
    device = torch.device("cuda")
    model = Model_V2(args.feature_size)
    test_loader = DataLoader(Dataset_Con_all_feedback_XD(args, test_mode=True), 
                            batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('/home/anas.al-lahham/AD_Unsupervised/unsupervised_ckpt/XDfinal.pkl').items()})
    scores = test(test_loader, model, args, device)
    