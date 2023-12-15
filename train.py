import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
import option
import math 
from tqdm import tqdm

args = option.parser.parse_args()
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCELoss()
# loss_fn = torch.nn.HingeEmbeddingLoss()


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def _init_(self):
        super(SigmoidMAELoss, self)._init_()
        from torch.nn import Sigmoid
        self._sigmoid_ = Sigmoid()
        self._l1_loss_ = MSELoss()

    def forward(self, pred, target):
        return self._l1_loss_(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def _init_(self):
        super(SigmoidCrossEntropyLoss, self)._init_()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))







def concatenated_train(loader, model, optimizer,device):
    with torch.set_grad_enabled(True):
        model.train()

        losses = []
        # for _ in range(30):  # 800/batch_size
        for _, (input, labels) in enumerate(loader):
            # input, labels = next(iter(loader)) 
            input, labels = input.to(device), labels.to(device)
            labels = labels.float()

            
            scores = model(input)
            scores = scores.float().flatten()
            loss = loss_fn(scores, labels)
            losses.append(loss.cpu().detach().numpy())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        return np.mean(losses)




def concatenated_train_top(loader, model, optimizer,device):
    with torch.set_grad_enabled(True):
        model.train()
        i= 0
        losses = []
        # for _ in range(30):  # 800/batch_size
        # for _ in range(100):
        for _, (input, labels) in enumerate(loader):
            
            # input, labels = next(iter(loader)) 
            input, labels = input.to(device), labels.to(device)
            labels = labels.float()
            # print(input.size())
            scores = model(input)
            # scores, feat_select_top, feat_select_low, top_select_score = model(input)
            scores = scores.squeeze()
            # print(scores.size())

            loss_criterion = loss_fn(scores, labels)

            # loss_sparse = sparsity(feat_select_top, 128, 8e-2) 
            # loss_smooth = smooth(top_select_score, 8e-3)

            loss = loss_criterion  #+ loss_sparse #+ loss_smooth #+loss_random(feat_select_low, feat_select_top) +los_clss(feat_select_low, feat_select_top)#+ 
            losses.append(loss.cpu().detach().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            # print(i)

        return np.mean(losses)




def concatenated_train_feedback(loader, model, optimizer, original_label, device):
    with torch.set_grad_enabled(True):
        model.train()

        losses = []
        original_labels = original_label
        new_labels = []
        import time
        start = time.time()
        for _, (input, idx) in enumerate(loader):
            labels = original_labels[idx]
            input, labels = input.to(device), labels.to(device)
            labels = labels.float()

            optimizer.zero_grad()
            # scores, feat_select_top, feat_select_low, top_select_score = model(input)
            scores = model(input)
            scores = scores.float().flatten()
            loss = loss_fn(scores, labels)
            # loss_sparse = sparsity(scores, 8e-3)
            # loss_smooth = smooth(scores, 8e-4)
            total_loss = loss # + loss_sparse + loss_smooth
            
            losses.append(total_loss.cpu().detach().numpy())
            trans = scores
            # trans = torch.where(scores > 0.6, 1.0, 0.0)
            # trans = (scores + labels)/2
            trans = trans.cpu().detach().numpy()
            res = list(zip(idx.cpu().numpy(), trans))
            new_labels += res

            total_loss.backward()
            optimizer.step()
        print(time.time() - start)
        return np.mean(losses), new_labels



   