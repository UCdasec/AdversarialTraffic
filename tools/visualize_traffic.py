import os
os.sys.path.append('..')

import torch
from train import models,utils_wf,utils_gan,utils_shs
from attacks.white_box.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
import numpy as np
import copy,sys



class traffic_plt:
    def __init__(self, opts, x_box_min=-1, x_box_max=0, pert_box=0.3):
        self.opts = opts
        self.mode = opts['mode']
        self.classifier_type = opts['classifier_type']
        self.model_path = '../model/' + self.mode + '/' + self.classifier_type
        self.pert_box = pert_box
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.input_nc, self.gen_input_nc = 1, 1

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        print('testing model path:  ', self.model_path)


    def model_reset(self,model):
        """
        given lstm model can't be backward in eval mode,
        so set dropout=0 and parameters require_grad=False in train mode, which is equal to eval mode
        """
        model_cp = copy.deepcopy(model)
        for p in model_cp.parameters():
            p.requires_grad = False

        return model_cp


    def ploting(self):

        "load data and model"
        if self.mode == 'wf':
            "load data"
            test_data = utils_wf.load_data_main(self.opts['test_data_path'], self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn_norm(params).to(self.device)

            elif self.classifier_type == 'lstm':
                params = utils_wf.params_lstm_eval(self.opts['num_class'], self.opts['input_size'],
                                                   self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)

        elif self.mode == 'shs':
            "load data"
            test_data = utils_shs.load_data_main(self.opts['test_data_path'], self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params(self.opts['num_class'], self.opts['input_size'])
            target_model = models.cnn_noNorm(params).to(self.device)

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()


        model_name = self.model_path + '/target_model.pth'
        target_model.load_state_dict(torch.load(model_name, map_location=self.device))

        "set equal_eval mode of train instead eval for lstm to aviod backward error"
        if self.classifier_type == 'lstm':
            target_model = self.model_reset(target_model)
            target_model.train()
        elif self.classifier_type == 'cnn':
            target_model.eval()

        target_model.to(self.device)


        ###############################################

        for i, data in enumerate(test_data):

            if i < self.opts['num_figs']:

                for Adversary in ['FGSM','PGD','DeepFool','GAN']:

                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)

                    "set adversary"
                    if self.mode == 'wf':
                        fgsm_epsilon = 0.1
                        pgd_a = 0.051
                    else:
                        fgsm_epsilon = 0.1
                        pgd_a = 0.01

                    if Adversary == 'GAN':
                        pretrained_generator_path = self.model_path + '/adv_generator.pth'
                        pretrained_G = models.Generator(self.gen_input_nc, self.input_nc).to(self.device)
                        pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=self.device))
                        pretrained_G.eval()
                    elif Adversary == 'FGSM':
                        adversary = FGSM(self.mode, self.x_box_min, self.x_box_max, self.pert_box, epsilon=fgsm_epsilon)
                    elif Adversary == 'DeepFool':
                        adversary = DeepFool(self.mode, self.x_box_min, self.x_box_max, self.pert_box, num_classes=5)
                    elif Adversary == 'PGD':
                        adversary = LinfPGDAttack(self.mode, self.x_box_min, self.x_box_max, self.pert_box, k=5,
                                                  a=pgd_a,
                                                  random_start=False)

                    "generate adversarial examples"
                    if Adversary == 'GAN':
                        pert = pretrained_G(x)
                        "cal adv_x given different mode wf/shs"
                        adv_x = utils_gan.get_advX_gan(x, pert, self.mode, pert_box=self.opts['pert_box'],
                                                       x_box_min=self.opts['x_box_min'],
                                                       x_box_max=self.opts['x_box_max'], alpha=self.opts['alpha'])

                    elif Adversary in ['FGSM', 'PGD', 'DeepFool']:
                        _, y_pred = torch.max(target_model(x), 1)

                        "cal adv_x given different mode wf/shs. the mode of adversary set before"
                        adversary.model = target_model
                        "use predicted label to prevent label leaking"
                        _, adv_x = adversary.perturbation(x, y_pred, self.opts['alpha'])

                    else:
                        print('No Adversary found! System will exit.')
                        sys.exit()

                    if self.mode == 'shs':
                        adv_x = (adv_x.data.cpu().numpy().squeeze() * 1500).round()
                    elif self.mode == 'wf':

                        "if the data use L2 normalized, then need to inverse it back"
                        # normalization = utils_wf.normalizer(x)
                        # adv_x = normalization.inverse_Normalizer(adv_x)

                        adv_x = adv_x.data.cpu().numpy().squeeze()
                        adv_x = adv_x.squeeze()
                    else:
                        print('mode should in ["wf","shs"], system will exit.')
                        sys.exit()


                    x_np = x.data.cpu().numpy().squeeze()
                    pert = adv_x - x_np

                    # traffics.append(np.squeeze(x.data.cpu().numpy().squeeze()))
                    # adv_traffics.append(adv_x)
                    # noise.append(pert)

                    "plot"
                    utils_shs.single_traffic_plot(self.classifier_type + '_'+ str(i), x_np, adv_x,Adversary)
                    utils_shs.noise_plot(self.classifier_type + '_'+ str(i), pert,Adversary)


def main(opts):
    traffic_plot = traffic_plt(opts, x_box_min=-1, x_box_max=0, pert_box=0.3)
    traffic_plot.ploting()


def get_opts_wf(num_figs,classifier_type):
    "parameters of website fingerprinting"

    return {
        'test_data_path': '../data/wf/test_NoDef_burst.csv',
        'classifier_type':classifier_type,
        'num_figs': num_figs,
        'mode': 'wf',
        'num_class': 95,
        'input_size': 512,
        'alpha': 10,
        'batch_size': 1,
        'pert_box':0.3,
        'x_box_min':-1,
        'x_box_max':1,

    }


def get_opts_shs(num_figs,classifier_type):
    "parameters of smart home speaker fingerprinting"

    return {
        'test_data_path': '../data/shs/traffic_test.csv',
        'classifier_type': classifier_type,
        'num_figs':num_figs,
        'mode': 'shs',
        'num_class': 101,
        'input_size': 256,
        'alpha': 1,
        'batch_size': 1,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 0,

    }



if __name__ == '__main__':

    mode = 'wf'                     # ['wf','shs']
    classifier_type = 'cnn'        # ['lstm','cnn']
    num_figs = 2

    if mode == 'wf':
        opts = get_opts_wf(num_figs,classifier_type)
    elif mode == 'shs':
        opts = get_opts_shs(num_figs,classifier_type)


    main(opts)