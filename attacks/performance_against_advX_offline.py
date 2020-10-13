"""
#1: test target model against adversarial examples of test data
#2: test target model with adversarial training against adversarial examples of test data
"""

import os
os.sys.path.append('..')

import torch
from train import utils_wf,utils_gan,utils_shs
from train import models
import sys


"""
adversarial examples of test data against target_model/adv_target_model
"""

class against_adv_x:
    def __init__(self,opts,x_box_min=-1,x_box_max=0,pert_box=0.3):
        self.opts = opts
        self.mode = opts['mode']
        self.adv_mode = opts['adv_mode']
        self.model_type = opts['model_type']
        self.classifier_type = opts['classifier_type']
        self.model_path = '../model/' + self.mode + '/' + self.classifier_type
        self.data_path = '../data/' + self.mode + '/' + self.classifier_type
        self.pert_box = pert_box
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.input_nc ,self.gen_input_nc = 1,1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf' or 'wf_ow' or 'wf_kf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        print('model path:  ',self.model_path)
        print('data path:   ',self.data_path)

    #------------------------------------------
    "performance of model"
    #------------------------------------------
    def test_model(self):

        "define test data type/path"
        # classifier cross validation:
        if self.opts['cross_validation']:
            if self.classifier_type == 'lstm':
                adv_test_data_path = '../data/' + self.mode + '/cnn/' + self.opts['adv_test_data_path']
            elif self.classifier_type == 'cnn':
                adv_test_data_path = '../data/' + self.mode + '/lstm/' + self.opts['adv_test_data_path']
            else:
                print('classifier type should in [lstm,cnn]. System will exit!')
                sys.exit()
        # test on Walkie-Talkie defended data
        elif self.opts['Adversary'] == 'WT':
            adv_test_data_path = '../data/WalkieTalkie/defended_csv/adv_test_WT.csv'
        else:
            adv_test_data_path = self.data_path + '/' + self.opts['adv_test_data_path']

        print('adv test data path:  ',adv_test_data_path)

        "load data and model"
        if self.mode == 'wf' or 'wf_ow' or 'wf_kf':
            "load data"
            if self.mode == 'wf_kf':
                test_path = '../data/wf/cross_val/' + self.opts['test_data_path']
                test_data = utils_wf.load_data_main(test_path,self.opts['batch_size'])
            else:
                test_path = '../data/wf/'+self.opts['test_data_path']
                test_data = utils_wf.load_data_main(test_path,self.opts['batch_size'])
            adv_test_data = utils_wf.load_data_main(adv_test_data_path,self.opts['batch_size'])
            print('test data path: ',test_path)

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn_norm(params).to(self.device)

            elif self.classifier_type == 'lstm':
                if self.mode == 'wf_ow':
                    params = utils_wf.params_lstm_ow_eval(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                else:
                    params = utils_wf.params_lstm_eval(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)

        elif self.mode == 'shs':
            "load data"
            test_data = utils_shs.load_data_main('../data/'+self.mode+'/'+self.opts['test_data_path'],self.opts['batch_size'])
            adv_test_data = utils_shs.load_data_main(adv_test_data_path,self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_shs.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn_noNorm(params).to(self.device)

            elif self.classifier_type == 'lstm':
                params = utils_shs.params_lstm_eval(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()


        if self.model_type == 'adv_target_model':
            if self.mode == 'wf_kf':
                model_name = self.model_path + '/adv_target_model_%s_%d.pth' % (self.opts['Adversary'],self.opts['id'])
            else:
                model_name = self.model_path + '/adv_target_model_' + self.opts['Adversary'] + '.pth'
        elif self.model_type == 'target_model':
            if self.mode == 'wf_kf':
                model_name = self.model_path + '/target_model_%d.pth' % self.opts['id']
            else:
                model_name = self.model_path + '/target_model.pth'
        else:
            print('target model type not in ["target_model","adv_target_model"], system will exit.')
            sys.exit()
        print('model path: ', model_name)

        target_model.load_state_dict(torch.load(model_name, map_location=self.device))
        target_model.eval()


        "test on adversarial examples"
        correct_adv_x = 0
        correct_x = 0
        total_case = 0
        for (x,y), (adv_x,adv_y) in zip(test_data,adv_test_data):

            x, y = x.to(self.device), y.to(self.device)
            adv_x,adv_y = adv_x.to(self.device), adv_y.to(self.device)

            "prediction on original input x"
            pred_x = target_model(x)
            _,pred_x = torch.max(pred_x, 1)
            correct_x += (pred_x == y).sum()

            "predition on adv_x"
            pred_adv_x = target_model(adv_x)
            _, pred_adv_x = torch.max(pred_adv_x, 1)
            correct_adv_x += (pred_adv_x == adv_y).sum()

            total_case += len(y)


        acc_x = float(correct_x.item()) / float(total_case)
        acc_adv = float(correct_adv_x.item()) / float(total_case)

        print('*'*30)
        print('"{}" with {} against {}.'.format(self.mode, self.opts['Adversary'], self.model_type))
        print('correct test after attack is {}'.format(correct_adv_x.item()))
        print('total test instances is {}'.format(total_case))
        print('accuracy of test after {} attack : correct/total= {:.5f}'.format(self.opts['Adversary'],acc_adv))
        print('success rate of the attack is : {}'.format(1 - acc_adv))
        print('accucary of the model without being attacked is {:.5f}'.format(acc_x))
        print('\n')


def main(opts):

    against_adv = against_adv_x(opts)
    against_adv.test_model()


def get_opts_wf(Adversary,mode,model_type,classifier_type,cross_validation):
    "parameters of website fingerprinting"
    return {
        'test_data_path': 'test_NoDef_burst.csv',
        'adv_test_data_path': 'adv_test_' + Adversary + '.csv',
        'model_type': model_type,
        'classifier_type': classifier_type,
        'cross_validation':cross_validation,
        'mode': mode,
        'adv_mode':'offline',
        'num_class': 95,
        'input_size': 512,
        'alpha': 10,
        'Adversary': Adversary,
        'batch_size': 64,
        'pert_box':0.3,
        'x_box_min':-1,
        'x_box_max':0,

    }


def get_opts_wf_kf(Adversary,mode,model_type,classifier_type,cross_validation,id):
    "parameters of website fingerprinting with 5-fold validation"
    return {
        'id': id,
        'test_data_path': 'traffic_test_%d.csv' % id,
        'adv_test_data_path': 'adv_test_%s_%d.csv' % (Adversary,id),
        'model_type': model_type,
        'classifier_type': classifier_type,
        'cross_validation':cross_validation,
        'mode': mode,
        'adv_mode':'offline',
        'num_class': 95,
        'input_size': 512,
        'alpha': 10,
        'Adversary': Adversary,
        'batch_size': 64,
        'pert_box':0.3,
        'x_box_min':-1,
        'x_box_max':0,

    }


def get_opts_wf_ow(Adversary,mode,model_type,classifier_type,cross_validation):
    "parameters of website fingerprinting"
    return {
        'test_data_path': 'test_NoDef_UnMon.csv',     # [UnMon, Mon]
        'adv_test_data_path': 'adv_test_UnMon_' + Adversary + '.csv',      # [UnMon, Mon]
        'model_type': model_type,
        'classifier_type': classifier_type,
        'cross_validation':cross_validation,
        'mode': mode,
        'adv_mode':'offline',
        'num_class': 96,
        'input_size': 512,
        'alpha': 10,
        'Adversary': Adversary,
        'batch_size': 64,
        'pert_box':0.3,
        'x_box_min':-1,
        'x_box_max':0,

    }


def get_opts_shs(Adversary,mode,model_type,classifier_type,cross_validation):
    "parameters of smart home speaker fingerprinting"
    return {
        'test_data_path': 'traffic_test.csv',
        'adv_test_data_path': 'adv_test_' + Adversary + '.csv',
        'model_type': model_type,
        'classifier_type': classifier_type,
        'cross_validation': cross_validation,
        'mode': mode,
        'adv_mode': 'offline',
        'num_class': 101,
        'input_size': 256,
        'alpha': 1,
        'Adversary': Adversary,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,

    }


if __name__ == '__main__':

    Adversary = ['DeepFool']    # 'FGSM','DeepFool','PGD','GAN', "WT" (walkietalkie)
    mode = 'wf_kf'                                # ['wf','shs','wf_ow','kf_kf]
    model_type = 'adv_target_model'            # ['target_model','adv_target_model': model with adv training]
    classifier_type = 'lstm'                    # ['lstm','cnn']
    cross_validation = False                     # classifier A against adv_x based on classifer B
    k = 5 # k-Fold validation

    for adv in Adversary:
        if mode == 'wf':
            opts = get_opts_wf(adv,mode,model_type,classifier_type,cross_validation)
        elif mode == 'wf_ow':
            opts = get_opts_wf_ow(adv,mode,model_type,classifier_type,cross_validation)
        elif mode == 'shs':
            opts = get_opts_shs(adv,mode,model_type,classifier_type,cross_validation)
        elif mode == 'wf_kf':
            for id in range(k):
                print('-'*30, '\n','%d fold....' % id)
                opts = get_opts_wf_kf(adv,mode,model_type,classifier_type,cross_validation,id)
                main(opts)

        if mode != 'wf_kf':
            main(opts)
