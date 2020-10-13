import os
os.sys.path.append('..')

from train import utils_wf
import torch
import numpy as np
import pandas as pd


"""
choose certain number cases from different adversearial examples csv files,
and append together to build train/test data for detecting adversary
"""

class file_build:
    def __init__(self,opts):

        self.opts = opts
        self.data_path = '../data/' + opts['mode'] + '/' + self.opts['classifier_type']
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')



    def select_file(self):

        X = pd.DataFrame([])
        Y = []

        for i,f in enumerate(self.opts['file_list']):
            x,y = utils_wf.load_csv_data(self.data_path+'/'+f)
            x = x.iloc[:self.opts['num_cases'],:]
            y = np.ones(self.opts['num_cases'],dtype=int) * i

            X = X.append(x,ignore_index=True)
            Y += list(y)

        return X,Y

    def write2csv(self,X,Y):
        X['label'] = Y
        X.to_csv(self.data_path + '/' + self.opts['output_file'],index=0)
        print('file has been processed successfully!')



def main(opts):
    get_file = file_build(opts)
    X,Y = get_file.select_file()
    get_file.write2csv(X,Y)


def get_opts():
    return{
        'mode': 'wf',
        'classifier_type':'cnn',
        'num_cases': 5000,
        # 'file_list': ['adv_test_FGSM.csv','adv_test_DeepFool.csv','adv_test_PGD.csv','adv_test_GAN.csv'],
        'file_list': ['adv_train_FGSM.csv','adv_train_DeepFool.csv','adv_train_PGD.csv','adv_train_GAN.csv'],
        'output_file': 'adv_train_all.csv',

    }

    #

if __name__ == '__main__':

    opts = get_opts()
    main(opts)



