import os
os.sys.path.append('..')

from train import utils_wf,utils_shs
from attacks.gan import gan
import sys
import time



def main(opts):

    mode = opts['mode']

    "load data"
    if mode in ['wf','wf_ow','wf_kf'] :
        "load data"
        print('train data: %s' % opts['train_data_path'])
        train_data = utils_wf.load_data_main(opts['train_data_path'], opts['batch_size'],shuffle=True)

    elif mode == 'shs':
        "load data"
        train_data = utils_shs.load_data_main(opts['train_data_path'], opts['batch_size'])

    else:
        print('mode not in ["wf","shs","wf_ow"], system will exit.')
        sys.exit()

    start_time = time.time()
    adv_gan = gan.advGan(opts,opts['x_box_min'],opts['x_box_max'],opts['pert_box'])
    adv_gan.train(train_data)
    end_time = time.time()
    print('training time of GAN is {} hours.'.format((end_time-start_time)/3600.0))



def get_opts_wf(mode,classifier_type,random_input_G):

    return {
        'epochs': 50,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,
        'alpha':10,
        'num_class': 95,
        'input_size': 512,
        'mode': mode,
        'random_input_G': random_input_G,
        'classifier_type':classifier_type,
        'train_data_path': '../data/wf/train_NoDef_burst.csv',
    }



def get_opts_wf_kf(mode,classifier_type,random_input_G,id):

    return {
        'id': id,
        'epochs': 50,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,
        'alpha':10,
        'num_class': 95,
        'input_size': 512,
        'mode': mode,
        'random_input_G': random_input_G,
        'classifier_type':classifier_type,
        'train_data_path': '../data/wf/cross_val/traffic_train_%d.csv' % id,
    }


def get_opts_wf_ow(mode,classifier_type,random_input_G):

    return {
        'epochs': 50,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,
        'alpha':10,
        'num_class': 96,
        'input_size': 512,
        'mode': mode,
        'random_input_G':random_input_G,
        'classifier_type':classifier_type,
        'train_data_path': '../data/wf_ow/train_NoDef_mix_gan.csv',
    }


def get_opts_shs(mode,classifier_type,random_input_G):
    "only in SHS, adv_x need to clamp to [-1,0] to maintain incoming traffic"
    return {
        'epochs': 50,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 0,
        'alpha':1,
        'num_class': 101,
        'input_size': 256,
        'mode': mode,
        'random_input_G':random_input_G,
        'classifier_type':classifier_type,
        'train_data_path': '../data/shs/traffic_train.csv',
    }



if __name__ == '__main__':

    mode = 'wf_kf'                # ['wf','shs','wf_ow','wf_kf']
    classifier_type = 'cnn'     #['lstm','cnn']
    random_input_G = True       # false: take orginal x as G's input, otherwise take random noise as G's input
    k = 5 #k-fold

    if mode == 'wf':
        opts = get_opts_wf(mode,classifier_type,random_input_G)
    elif mode == 'wf_ow':
        opts = get_opts_wf_ow(mode,classifier_type,random_input_G)
    elif mode == 'shs':
        opts = get_opts_shs(mode,classifier_type,random_input_G)
    elif mode == 'wf_kf':
        for id in range(1,2):
            print('%d Fold ...' % id)
            opts = get_opts_wf_kf(mode,classifier_type,random_input_G,id)
            main(opts)
        sys.exit('k-fold completed1, system outs!')

    else:
        print("mode shoud in ['wf','shs','wf_ow','wf_kf'], system will exit.")
        sys.exit()

    main(opts)
