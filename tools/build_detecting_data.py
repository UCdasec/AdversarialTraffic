import os
os.sys.path.append('..')
import torch
import pandas as pd
import numpy as np
from train import utils_wf

"""
select certain cases per class per Adversary
"""
class get_detect_data:
    def __init__(self,opts):
        self.opts = opts
        self.data_path = '../data/' + opts['mode'] + '/' + self.opts['classifier_type']
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')



    def select_file(self):

        X = []
        Y = []

        for i,f in enumerate(self.opts['file_list']):

            x, _ = utils_wf.extract_data_each_class(self.data_path+'/'+f, self.opts['num_cases'])
            y = np.ones(len(x), dtype=int) * i

            X += x
            Y += list(y)

        return X,Y


    def write2csv(self,X,Y):

        df = utils_wf.convert2dataframe(X,Y,mode='NoPadding')
        df.to_csv(self.data_path + '/' + self.opts['output_file'],index=0)
        print('file has been processed successfully!')


def main(opts):
    get_file = get_detect_data(opts)
    X,Y = get_file.select_file()
    get_file.write2csv(X,Y)


def get_opts():
    return{
        'mode': 'wf',
        'classifier_type':'cnn',
        'num_cases': 200,   # train:300, test:50
        # 'file_list': ['adv_test_FGSM.csv','adv_test_DeepFool.csv','adv_test_PGD.csv','adv_test_GAN.csv'],
        'file_list': ['adv_train_FGSM.csv','adv_train_DeepFool.csv','adv_train_PGD.csv','adv_train_GAN.csv'],
        'output_file': 'adv_train_all.csv',

    }



if __name__ == '__main__':

    opts = get_opts()
    main(opts)