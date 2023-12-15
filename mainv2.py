from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utillsv2 import Concat_list_all_crop_feedback
from model import Model, Model_V2
from dataset import  Dataset_Con_all_feedback_XD
from train import concatenated_train, concatenated_train_feedback
from test import test
import option
from tqdm import tqdm

import os
import numpy as np
import wandb
import copy
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    print('mainv2')
    args = option.parser.parse_args()

    len_N, original_lables  = Concat_list_all_crop_feedback(Test=False, create='False')
    wandb.login()
    wandb.init(project="Unsupervised Anomaly Detection", config=args)

    test_loader = DataLoader(Dataset_Con_all_feedback_XD(args, test_mode=True), 
                            batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    
    
    train_loader = DataLoader(Dataset_Con_all_feedback_XD(args, test_mode=False, is_normal=True), 
                                batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.workers, pin_memory=False)
    
    model = Model_V2(args.feature_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    test_info = {"epoch": [], "test_auc": []}

    auc, ap = test(test_loader, model, args, device)
    print("epcoh 0 auc = ", auc)
    wandb.log({'AUC': auc,'AP': ap}, step=0)

    for epoch in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):
        loss, lls = concatenated_train_feedback(train_loader, model, optimizer,original_lables, device )
        auc, ap = test(test_loader, model, args, device)
        test_info["epoch"].append(epoch)
        test_info["test_auc"].append(auc)
        scheduler.step()
        
        update = sorted(lls, key=lambda x: x[0])

        print('\nEpoch {}/{}, LR: {:.4f} auc: {:.4f}, ap: {:.4f}, loss: {:.4f}\n'.format(epoch, args.max_epoch, optimizer.param_groups[0]['lr'] , auc, ap, loss))
        wandb.log({'AUC': auc,'AP': ap, 'loss': loss}, step=(epoch+1)*544)

wandb.run.name = args.datasetname
torch.save(model.state_dict(), './unsupervised_ckpt/' + args.datasetname + 'final.pkl')


        
