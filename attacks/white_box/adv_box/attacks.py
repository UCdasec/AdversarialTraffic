"""
white box attacks, include FGSM, DeepFool and PGD, for generating adversarial examples and
specifically customized for websites network taffic 1D dataset.

customized parameter alpha = 10 added , and is for the traffic trace in burst level, users can change it
based on your requirements

"""

import os
os.sys.path.append('..')

from train import utils_wf
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import copy, sys





class FGSM(object):

    def __init__(self, mode,x_box_min=-1,x_box_max=0,pert_box=0.3,model=None, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon
        self.mode = mode
        self.pert_box = pert_box
        self.x_box_min, self.x_box_max = x_box_min,x_box_max
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def perturbation(self, X_i, y, alpha=10):
        """
        given examples (X,y), returns corresponding adversarial example
        y should be a integer
        """

        X = np.copy(X_i.cpu())  #the input X_i is a tensor, so before copy, need to copy it in cpu; array
        X = torch.from_numpy(X)

        X_var = Variable(X.to(self.device), requires_grad=True)
        y_var = Variable(y.to(self.device))

        output = self.model(X_var)
        loss = self.criterion(output, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()  # convert cuda/cpu data to numpy (array)
        pert = torch.from_numpy(self.epsilon * grad_sign)

        "adjust perturbation given the mode=['wf','shs']"
        if self.mode == 'wf' or 'wf_ow':

            X = utils_wf.get_advX_wf_main(X,pert,self.pert_box,alpha)
            X = X.to(self.device)

        elif self.mode == 'shs':
            X += pert

            "clamp generated adv example in a certain range"
            X = torch.clamp(X, self.x_box_min, self.x_box_max)
        else:
            print('mode should in ["wf","shs"], system will exit.')
            sys.exit()

        "get label for generated adv example"
        output_pert = self.model(X.to(self.device))
        _,y_pert = torch.max(output_pert.data,1)

        return y_pert, X    # X in numpy format



class DeepFool(object):

    """
    NOTE: the orginal code from the paper (deepfool paper), may have errors in terms of preds_label_sort
    they did not iterate parameter "preds_label_sort", instead they only computed it
    once at the very first begin based on the original input X_input
    we correct it by add "preds_label_sort" to the iteration step based on X_input first
    and then iterately based on perturbed x

    """

    def __init__(self,mode,x_box_min=-1,x_box_max=0,pert_box=0.3,model=None, num_classes=5):
        """
        num_classes: limits the number of classes to test against
        overshoot: used as a termination criterion to prevent vanishing updates
        max_iter: maximum number of iterations for deepfool
        return: perturbed output
        """
        self.model = model
        self.num_classes = num_classes
        self.mode = mode
        self.pert_box = pert_box
        self.x_box_min, self.x_box_max = x_box_min,x_box_max
        self.overshoot = 0.02
        self.max_iter = 20  #default 50
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def perturbation(self, X_input,y,alpha=10):
        "y: label is not needed here"

        X_var = Variable(X_input.to(self.device),requires_grad=True)
        preds = self.model(X_var)

        preds_label_sort = torch.argsort(preds, descending=True)  # return index of sorted value in descending direction, compatable to mini batch
        preds_label_sort = preds_label_sort[:, 0:self.num_classes]
        pred_label = preds_label_sort[:, 0]

        input_shape = X_input.cpu().numpy().shape
        w = np.zeros(input_shape)
        noise_total = np.zeros(input_shape)
        pert_x = copy.deepcopy(X_input)
        loop_i = 0                  # number of iteratioins applied
        pert_label_i = pred_label   # in the end, pert_label_i refer to the perturbed label corresponding to perturbed_x

        while pert_label_i == pred_label and loop_i < self.max_iter:

            pert = np.inf     # noise at each step, format: infinity float
            preds[:,preds_label_sort[:,0]].backward(retain_graph=True)
            grad_orig = X_var.grad.data.cpu().numpy().copy()

            for k in range(1,self.num_classes):

                #back propagation
                zero_gradients(X_var)

                preds[:,preds_label_sort[:,k]].backward(retain_graph=True)
                cur_grad = X_var.grad.data.cpu().numpy().copy()

                #set new w_k and new preds_k
                w_k = cur_grad - grad_orig
                preds_k = (preds[:,preds_label_sort[:,k]] - preds[:,preds_label_sort[:,0]]).data.cpu().numpy()

                # Normalization: ||w_k||**2,
                w_k_norm = np.linalg.norm(w_k.flatten())

                # smoothing avoid to divide by zero
                if w_k_norm == 0:
                    overshoot_noise = 1e-4   # the smaller, the higher of 'pert_k', will be dropped during the iterations
                    pert_k = abs(preds_k) / (overshoot_noise + w_k_norm)
                else:
                    pert_k = abs(preds_k) / w_k_norm

                # determin which w_k to use, select the smalles one.
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # calculate noise_i and noise_total
            # add 1e-4 for numerical stability
            w_norm = np.linalg.norm(w)
            if w_norm == 0:
                noise_i = (pert + 1e-4) * w / (w_norm + 1)  # add 1, avoid divide by zero
            else:
                noise_i = (pert + 1e-4) * w / w_norm

            noise_total = np.float32(noise_total + noise_i)


            "get pertburbation"
            noise_total_overshoot = ((1 + self.overshoot) * torch.from_numpy(noise_total)).to(self.device)

            "adjust perturbation given the mode=['wf','shs']"
            if self.mode == 'wf' or 'wf_ow':
                pert_x = utils_wf.get_advX_wf_main(X_input,noise_total_overshoot,self.pert_box,alpha)
                pert_x = pert_x.to(self.device)

            elif self.mode == 'shs':
                pert_x = X_input + noise_total_overshoot

                "clamp adv example to certain range"
                pert_x = torch.clamp(pert_x, self.x_box_min, self.x_box_max)
            else:
                print('mode should in ["wf","shs"], system will exit.')
                sys.exit()

            "get lable for adv example"
            X_var = Variable(pert_x.to(self.device), requires_grad = True)
            preds = self.model(X_var)
            preds_label_sort = torch.argsort(preds, descending=True)  # return index of sorted value in descending direction in orginal shape, compatable to mini batch
            preds_label_sort = preds_label_sort[:, 0:self.num_classes]
            pert_label_i = preds_label_sort[:,0]

            loop_i += 1

        noise_total = (1 + self.overshoot) * noise_total    # minimal pertubation that fools the classifier

        return pert_label_i, pert_x       #return pert label and pert x




class LinfPGDAttack(object):
    def __init__(self, mode, x_box_min=-1,x_box_max=0,pert_box=0.3,model=None, epsilon=0.3, k=40, a=0.01,
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.mode = mode
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pert_box = pert_box
        self.x_box_min, self.x_box_max = x_box_min, x_box_max

    def perturbation(self, X_nat, y,alpha=10):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            X = X_nat + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')).to(self.device)
        else:
            X = copy.deepcopy(X_nat)


        for i in range(self.k):
            X_var = Variable(X.to(self.device), requires_grad=True)
            # y_var = Variable(torch.LongTensor(y).to(self.device))
            y_var = Variable(y.to(self.device))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad_sign = X_var.grad.data.cpu().sign().numpy()
            pert = torch.from_numpy(self.a * grad_sign).to(self.device)

            "adjust perturbation given the mode=['wf','shs']"
            if self.mode == 'wf' or 'wf_ow':

                X = utils_wf.get_advX_wf_main(X_var,pert,self.pert_box,alpha)
                X = X.to(self.device)

            elif self.mode == 'shs':
                X += pert
                X = torch.clamp(X, self.x_box_min, self.x_box_max)
            else:
                print('mode should in ["wf","shs"], system will exit.')
                sys.exit()

        y_adv = self.model(X.to(self.device))
        _,y_adv = torch.max(y_adv,1)

        return y_adv, X
