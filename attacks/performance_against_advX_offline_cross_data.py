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

        if self.mode == 'wf':
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

        "classifier cross validation: define test data type"
        if self.opts['cross_val_classifier']:
            if self.classifier_type == 'lstm':
                adv_test_data_path = '../data/' + self.mode + '/cnn/' + self.opts['adv_test_data_path']
            elif self.classifier_type == 'cnn':
                adv_test_data_path = '../data/' + self.mode + '/lstm/' + self.opts['adv_test_data_path']
            else:
                print('classifier type should in [lstm,cnn]. System will exit!')
                sys.exit()
        elif self.opts['cross_val_dataset']:
            adv_test_data_path = self.data_path + '/' + self.opts['adv_test_data_path_cross_dataset']
        else:
            adv_test_data_path = self.data_path + '/' + self.opts['adv_test_data_path']

        print('adv test data path:  ',adv_test_data_path)

        "load data and model"
        if self.mode == 'wf':
            "load data"
            test_data = utils_wf.load_data_main('../data/'+self.mode+'/'+self.opts['test_data_path'],self.opts['batch_size'])
            adv_test_data = utils_wf.load_data_main(adv_test_data_path,self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn_norm(params).to(self.device)

            elif self.classifier_type == 'lstm':
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
            model_name = self.model_path + '/adv_target_model_' + self.opts['Adversary'] + '.pth'
        elif self.model_type == 'target_model':
            model_name = self.model_path + '/target_model.pth'
        else:
            print('target model type not in ["target_model","adv_target_model"], system will exit.')
            sys.exit()

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
        # print('"{}" with {} against {}.'.format(self.mode, self.opts['Adversary'], self.model_type))
        print('correct test after attack is {}'.format(correct_adv_x.item()))
        print('total test instances is {}'.format(total_case))
        print('accuracy of test after attack : correct/total= {:.5f}'.format(acc_adv))
        print('success rate of the attack is : {}'.format(1 - acc_adv))
        print('accucary of the model without being attacked is {:.5f}'.format(acc_x))
        print('\n')


def main(opts):

    """
    cross_data, model adv train on adv_x generated by A,
    test on adv_x generated by adversary B

    """
    adversaries = ['FGSM', 'DeepFool','PGD', 'GAN', ]
    if opts['cross_val_dataset']:
        # adversaries.remove(opts['Adversary'])
        for adv in adversaries:

            print('cross data validation: ', opts['cross_val_dataset'])
            print('adv training with: ', opts['Adversary'] )
            print('adv_train_model against: ', adv)

            opts['adv_test_data_path_cross_dataset'] = 'adv_test_' + adv + '.csv'
            against_adv = against_adv_x(opts)
            against_adv.test_model()



def get_opts_wf(Adversary,model_type,classifier_type,cross_val_classifier,cross_val_dataset):
    "parameters of website fingerprinting"
    return {
        'test_data_path': 'test_NoDef_burst.csv',
        'adv_test_data_path': 'adv_test_' + Adversary + '.csv',
        'adv_test_data_path_cross_dataset': '',
        'model_type': model_type,
        'classifier_type': classifier_type,
        'cross_val_classifier':cross_val_classifier,
        'cross_val_dataset': cross_val_dataset,
        'mode': 'wf',
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


def get_opts_shs(Adversary,model_type,classifier_type,cross_val_classifier,cross_val_dataset):
    "parameters of smart home speaker fingerprinting"
    return {
        'test_data_path': 'traffic_test.csv',
        'adv_test_data_path': 'adv_test_' + Adversary + '.csv',
        'model_type': model_type,
        'classifier_type': classifier_type,
        'cross_val_classifier': cross_val_classifier,
        'cross_val_dataset': cross_val_dataset,
        'mode': 'shs',
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

    Adversary = ['GAN',]
    mode = 'wf'                                # ['wf','shs']
    model_type = 'adv_target_model'            # ['target_model','adv_target_model']
    classifier_type = 'lstm'                    # ['lstm','cnn']
    cross_val_classifier = False                     # classifier A against adv_x based on classifer B
    cross_val_dataset = True                    # adv training on adv_x based on adversary A, against adv_x based on adversary B


    for adv in Adversary:
        print('adversary: ', adv)
        if mode == 'wf':
            opts = get_opts_wf(adv, model_type, classifier_type, cross_val_classifier, cross_val_dataset)
        elif mode == 'shs':
            opts = get_opts_shs(adv, model_type, classifier_type, cross_val_classifier, cross_val_dataset)


        main(opts)
