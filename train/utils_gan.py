import os
os.sys.path.append('..')

import torch
from train import utils_wf
import sys




def get_advX_gan(x,pert,mode,pert_box=0.3,x_box_min=-1,x_box_max=0,alpha=None):
    """
    given different mode, compute the adv_x
    x: torch.Tensor
    pert: torch.Tensor
    mode: ['wf','shs']
    return: torch.Tensor. adversarial example,
    """
    if mode == 'wf' or 'wf_ow':

        alpha = (alpha) * 2500  # for lstm
        # alpha = (alpha+10) * 5000  # for cnn
        # alpha = 500   # for L2 normalized data

        adv_x = utils_wf.get_advX_wf_main(x,pert,pert_box,alpha)
    elif mode == 'shs':
        pert = torch.clamp(pert, -pert_box, pert_box)
        adv_x = pert + x
        adv_x = torch.clamp(adv_x, x_box_min, x_box_max)
    else:
        print('mode should in ["wf","shs"], system will exit.')
        sys.exit()

    return adv_x
