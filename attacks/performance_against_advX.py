"""
#1: test target model against adversarial examples of test data
#2: test target model with adversarial training against adversarial examples of test data
"""

import os
os.sys.path.append('..')

import torch
from attacks.white_box.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
from train import utils_wf,utils_gan,utils_shs
from train import models
import sys,copy


"""
adversarial examples of test data against target_model/adv_target_model
"""

class against_adv_x:
    def __init__(self,opts,x_box_min=-1,x_box_max=0,pert_box=0.3):
        self.opts = opts
        self.mode = opts['mode']
        self.target_model_type = opts['target_model_type']
        self.classifier_type = opts['classifier_type']
        self.model_path = '../model/' + self.mode + '/' + opts['classifier_type']
        self.pert_box = pert_box
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.input_nc ,self.gen_input_nc = 1,1

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...with {}'.format(self.classifier_type))
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...with {}'.format(self.classifier_type))

        # if not os.path.exists(self.model_path):
        #     os.makedirs(self.model_path)
        print('testing model path:  ',self.model_path)


    def model_reset(self,model):
        """
        given lstm model can't be backward in eval mode,
        so set dropout=0 and parameters require_grad=False in train mode, which is equal to eval mode
        """
        model_cp = copy.deepcopy(model)
        for p in model_cp.parameters():
            p.requires_grad = False

        return model_cp


    #------------------------------------------
    "performance of model"
    #------------------------------------------
    def test_model(self):

        "load data and model"
        if self.mode == 'wf':
            "load data"
            test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn_norm(params).to(self.device)

            elif self.classifier_type == 'lstm':
                params = utils_wf.params_lstm_eval(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)


        elif self.mode == 'shs':
            "load data"
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.cnn_noNorm(params).to(self.device)

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        if self.target_model_type == 'adv_target_model':
            model_name = self.model_path + '/adv_target_model_' + self.opts['Adversary'] + '.pth'
        elif self.target_model_type == 'target_model':
            model_name = self.model_path + '/target_model.pth'
        else:
            print('target model type not in ["target_model","adv_target_model"], system will exit.')
            sys.exit()

        target_model.load_state_dict(torch.load(model_name, map_location=self.device))

        "set equal_eval mode of train instead eval for lstm"
        if self.classifier_type == 'lstm':
            target_model = self.model_reset(target_model)
            target_model.train()
        elif self.classifier_type == 'cnn':
            target_model.eval()

        target_model.to(self.device)


        "set adversary"
        Adversary = self.opts['Adversary']

        if self.mode == 'wf':
            fgsm_epsilon = 0.1
            pgd_a = 0.051
        elif self.mode == 'shs':
            fgsm_epsilon = 0.1
            pgd_a = 0.01

        if Adversary == 'GAN':
            pretrained_generator_path = self.model_path + '/adv_generator.pth'
            pretrained_G = models.Generator(self.gen_input_nc, self.input_nc).to(self.device)
            pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=self.device))
            pretrained_G.eval()
        elif Adversary == 'FGSM':
            adversary = FGSM(self.mode, self.x_box_min, self.x_box_max,self.pert_box, epsilon=fgsm_epsilon)
        elif Adversary == 'DeepFool':
            adversary = DeepFool(self.mode, self.x_box_min, self.x_box_max,self.pert_box, num_classes=5)
        elif Adversary == 'PGD':
            adversary = LinfPGDAttack(self.mode, self.x_box_min, self.x_box_max,self.pert_box, k=5, a=pgd_a, random_start=False)


        "test on adversarial examples"
        num_correct = 0
        correct_x = 0
        total_case = 0
        for i, data in enumerate(test_data, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)

            "prediction on original input x"
            pred_y = target_model(test_x)
            _,pred_y = torch.max(pred_y, 1)

            "prediction on adversarial x"
            if Adversary in ['FGSM', 'DeepFool', 'PGD']:
                adversary.model = target_model

                "use predicted label to prevent label leaking"
                adv_y,adv_x = adversary.perturbation(test_x,pred_y,self.opts['alpha'])
                # adv_y,adv_x = adversary.perturbation(test_x,test_y,self.opts['alpha'])

            elif Adversary == 'GAN':
                pert = pretrained_G(test_x)
                adv_x = utils_gan.get_advX_gan(test_x,pert,self.mode,self.pert_box,self.x_box_min,self.x_box_max,self.opts['alpha'])
                adv_y = torch.argmax(target_model(adv_x.to(self.device)),1)

            num_correct += torch.sum(adv_y == test_y, 0)
            correct_x += (pred_y == test_y).sum()
            total_case += len(test_y)

        acc = float(num_correct.item()) / float(total_case)

        print('*'*30)
        print('"{}" with {} against {}.'.format(self.mode,Adversary,self.target_model_type) )
        print('correct test after attack is {}'.format(num_correct.item()))
        print('total test instances is {}'.format(total_case))
        print('accuracy of test after {} attack : correct/total= {:.5f}'.format(Adversary,acc))
        print('success rate of the attack is : {}'.format(1 - acc))
        print('accucary of the model without being attacked is {:.5f}'.format(float(correct_x) / float(total_case)))
        print('\n')


def main(opts):

    against_adv = against_adv_x(opts)
    against_adv.test_model()


def get_opts_wf(Adversary,model_type,classifier_type):
    "parameters of website fingerprinting"
    return {
        'test_data_path': '../data/wf/test_NoDef_burst.csv',
        'target_model_type': model_type,
        'classifier_type':classifier_type,
        'mode': 'wf',
        'num_class': 95,
        'input_size': 512,
        'alpha': 10,
        'Adversary': Adversary,
        'batch_size': 64,
        'pert_box':0.3,
        'x_box_min':-1,
        'x_box_max':1,

    }


def get_opts_shs(Adversary,model_type,classifier_type):
    "parameters of smart home speaker fingerprinting"
    return {
        'test_data_path': '../data/shs/traffic_test.csv',
        'target_model_type': model_type,
        'classifier_type': classifier_type,
        'mode': 'shs',
        'num_class': 101,
        'input_size': 256,
        'alpha': 1,
        'Adversary': Adversary,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 0,

    }


if __name__ == '__main__':

    Adversary = ['FGSM','GAN','PGD','DeepFool',]

    mode = 'wf'                     # ['wf','shs']
    model_type = 'target_model'     # "model type includes: ['target_model','adv_target_model']"
    classifier_type = 'cnn'        # ['lstm','cnn']

    for adv in Adversary:
        if mode == 'wf':
            opts = get_opts_wf(adv,model_type,classifier_type)
        elif mode == 'shs':
            opts = get_opts_shs(adv,model_type,classifier_type)

        "set batch_szie, deepfool only work at batch_size=1"
        if adv == 'DeepFool':
            opts['batch_size'] = 1
            print('batch_size {}'.format(opts['batch_size']))
        else:
            opts['batch_size'] = 1
            print('batch_size {}'.format(opts['batch_size']))
            pass

        main(opts)
