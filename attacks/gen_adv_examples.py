"generate adversarial examples of FGSM/PGD/DeepFool. GAN not include yet"

import os
os.sys.path.append('..')

import torch
import pandas as pd
from attacks.white_box.adv_box.attacks import FGSM, DeepFool, LinfPGDAttack
from train import models,utils_wf,utils_shs,utils_gan
import sys,copy
import time



class gen_adv_x:
    def __init__(self,opts, x_box_min=-1,x_box_max=0,pert_box=0.3):

        self.opts = opts
        self.mode = opts['mode']
        self.classifier_type = opts['classifier_type']
        self.model_path = '../model/' + self.mode + '/' + opts['classifier_type']
        self.data_path = '../data/' + self.mode + '/' + opts['classifier_type']
        self.pert_box = pert_box
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.input_nc, self.gen_input_nc = 1, 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf' or 'wf_ow' or 'wf_kf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        "creat data folder"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)


    def x_adv_gen(self,x, y, model, adversary):
        """
        for white-box based adv training, target model training should separate with adv examples generation.
        cause adv_training related to target model, and adv example generation involved target model.
        therefore, should set target model as evaluation mode when producing adv examples in adv training
        """
        model_cp = copy.deepcopy(model)
        for p in model_cp.parameters():
            p.requires_grad = False

        if self.classifier_type == 'cnn':
            model_cp.eval()
        elif self.classifier_type == 'lstm':
            model_cp.train()

        adversary.model = model_cp
        _,x_adv = adversary.perturbation(x, y,self.opts['alpha'])

        return x_adv


    def get_adv_x(self, data, adv_name,adv_model,target_model,input_data_type):
        "gen adv_x for data given Adversary"
        adv_xs = []
        labels = []
        start_time = time.time()
        for i, (x, y) in enumerate(data):
            print('{} generate adversary example of data {} ...'.format(adv_name, i))

            x, y = x.to(self.device), y.to(self.device)


            if adv_name == 'GAN':
                pert = adv_model(x)
                "cal adv_x given different mode wf/shs"
                if self.mode == 'wf_ow':
                    alpha = self.opts['alpha']*1
                else:
                    alpha = self.opts['alpha']
                adv_x = utils_gan.get_advX_gan(x, pert, self.mode, pert_box=self.opts['pert_box'],
                                               x_box_min=self.opts['x_box_min'],
                                               x_box_max=self.opts['x_box_max'], alpha=alpha)
            elif adv_name in ['FGSM', 'PGD', 'DeepFool']:
                _, y_pred = torch.max(target_model(x), 1)

                "cal adv_x given different mode wf/shs. the mode of adversary set before"
                "use predicted label to prevent label leaking"
                adv_x = self.x_adv_gen(x, y_pred, target_model, adv_model)

            else:
                print('No Adversary found! System will exit.')
                sys.exit()

            if self.mode == 'shs':
                adv_x = (adv_x.data.cpu().numpy().squeeze() * 1500).round()
            elif self.mode == 'wf' or 'wf_ow' or 'wf_kf':

                "if the data use L2 normalized, then need to inverse it back"
                # normalization = utils_wf.normalizer(x)
                # adv_x = normalization.inverse_Normalizer(adv_x)

                adv_x = adv_x.data.cpu().numpy().squeeze()
                adv_x = adv_x.squeeze()
            else:
                print('mode should in ["wf","shs","wf_ow","wf_kf"], system will exit.')
                sys.exit()

            adv_xs.append(adv_x)
            labels.append(y.data.cpu().numpy().squeeze())

        end_time = time.time()
        ave_gen_time = (end_time - start_time) / float(i+1)
        print('generation time with {} is {} seconds'.format(adv_name, ave_gen_time))

        # convert to dataframe
        adv_xs = pd.DataFrame(adv_xs)
        adv_xs['label'] = labels

        # get output name
        if self.mode == 'wf_ow':
            if input_data_type == 'test':
                input_data_type = 'test_Mon'
        if self.mode == 'wf_kf':
            output_path= self.data_path + '/adv_' + input_data_type + '_' + self.opts['Adversary'] + '_%d.csv' % self.opts['id']
        else:
            output_path = self.data_path + '/adv_' + input_data_type + '_' + self.opts['Adversary'] + '.csv'

        # save data to csv
        adv_xs.to_csv(output_path, index=0)
        print('adversary examples of {} data is generated'.format(input_data_type))



    def generate(self):
        "generate adv_x given x, append with its original label y instead with y_pert "

        "load data and target model"
        if self.mode in ['wf', 'wf_ow', 'wf_kf','wf_sg']:

            "load data"
            if self.mode != 'wf_sg':
                train_data = utils_wf.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
                test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])
            else:
                input_data = utils_wf.load_data_main(self.opts['input_data_path'],self.opts['batch_size'])
                train_data, test_data = []
            if self.mode == 'wf_ow':
                test_data_UnMon = utils_wf.load_data_main(self.opts['test_data_path_UnMon'], self.opts['batch_size'])

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
            train_data = utils_shs.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_shs.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn_noNorm(params).to(self.device)

            elif self.classifier_type == 'lstm':
                params = utils_shs.params_lstm_eval(self.opts['num_class'], self.opts['input_size'],self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)


        else:
            print('mode not in ["wf","shs","wf_ow","wf_kf"], system will exit.')
            sys.exit()

        # load trained model
        if self.mode == 'wf_kf':
            model_name = '/target_model_%d.pth' % self.opts['id']
        else:
            model_name = '/target_model.pth'
        model_path = self.model_path + model_name
        target_model.load_state_dict(torch.load(model_path, map_location=self.device))

        "set adversary"
        Adversary = self.opts['Adversary']

        if self.mode == 'wf' or 'wf_ow' or 'wf_kf':
            fgsm_epsilon = 0.1
            pgd_a = 0.051       # if data un-normalized
            # pgd_a = 0.01    # if data normalized
        else:
            fgsm_epsilon = 0.1
            pgd_a = 0.01

        if Adversary == 'GAN':
            if self.mode == 'wf_kf':
                g_name = '/adv_generator_%d.pth' % self.opts['id']
            else:
                g_name = '/adv_generator.pth'
            pretrained_generator_path = self.model_path + g_name
            adversary = models.Generator(self.gen_input_nc, self.input_nc).to(self.device)
            adversary.load_state_dict(torch.load(pretrained_generator_path, map_location=self.device))
            adversary.eval()
        elif Adversary == 'FGSM':
            adversary = FGSM(self.mode, self.x_box_min, self.x_box_max, self.pert_box, epsilon=fgsm_epsilon)
        elif Adversary == 'DeepFool':
            adversary = DeepFool(self.mode, self.x_box_min, self.x_box_max, self.pert_box, num_classes=5)
        elif Adversary == 'PGD':
            adversary = LinfPGDAttack(self.mode, self.x_box_min, self.x_box_max, self.pert_box, k=5,a=pgd_a, random_start=False)


        "produce adv_x given differen input data"
        # test data Monitored
        data_type = 'test'
        self.get_adv_x(test_data,Adversary,adversary,target_model,data_type)


        # test data Unmonitored in open world setting
        if self.mode == 'wf_ow':
            data_type = 'test_UnMon'
            self.get_adv_x(test_data_UnMon, Adversary, adversary, target_model,data_type)

        #train data
        data_type = 'train'
        self.get_adv_x(train_data, Adversary, adversary, target_model,data_type)



        # "gen adv_x for test data"
        # adv_xs = []
        # labels = []
        # for i,(x,y) in enumerate(test_data):
        #     print('{} generate adversary example of test data {} ...'.format(Adversary,i))
        #
        #     x, y = x.to(self.device), y.to(self.device)
        #
        #     if Adversary == 'GAN':
        #         pert = pretrained_G(x)
        #         "cal adv_x given different mode wf/shs"
        #         adv_x = utils_gan.get_advX_gan(x, pert, self.mode, pert_box=self.opts['pert_box'],
        #                                        x_box_min=self.opts['x_box_min'],
        #                                        x_box_max=self.opts['x_box_max'], alpha=self.opts['alpha'])
        #     elif Adversary in ['FGSM', 'PGD','DeepFool']:
        #         _, y_pred = torch.max(target_model(x), 1)
        #
        #         "cal adv_x given different mode wf/shs. the mode of adversary set before"
        #         "use predicted label to prevent label leaking"
        #         adv_x = self.x_adv_gen(x, y_pred, target_model, adversary)
        #
        #     else:
        #         print('No Adversary found! System will exit.')
        #         sys.exit()
        #
        #
        #     if self.mode == 'shs':
        #         adv_x = (adv_x.data.cpu().numpy().squeeze() * 1500).round()
        #     elif self.mode == 'wf':
        #
        #         "if the data use L2 normalized, then need to inverse it back"
        #         # normalization = utils_wf.normalizer(x)
        #         # adv_x = normalization.inverse_Normalizer(adv_x)
        #
        #         adv_x = adv_x.data.cpu().numpy().squeeze()
        #         adv_x = adv_x.squeeze()
        #     else:
        #         print('mode should in ["wf","shs"], system will exit.')
        #         sys.exit()
        #
        #     adv_xs.append(adv_x)
        #     labels.append(y.data.cpu().numpy().squeeze())
        #
        # adv_xs = pd.DataFrame(adv_xs)
        # adv_xs['label'] = labels
        # output_path = self.data_path + '/adv_test_' + self.opts['Adversary'] + '.csv'
        # adv_xs.to_csv(output_path,index=0)
        # print('adversary examples of test data is generated')
        #
        #
        # "gen adv_x for train data"
        # adv_xs = []
        # labels = []
        # for i,(x,y) in enumerate(train_data):
        #
        #     print('{} generate adversary example of train data {} ...'.format(Adversary,i))
        #
        #     x, y = x.to(self.device), y.to(self.device)
        #
        #     if Adversary == 'GAN':
        #         pert = pretrained_G(x)
        #         "cal adv_x given different mode wf/shs"
        #         adv_x = utils_gan.get_advX_gan(x, pert, self.mode, pert_box=self.opts['pert_box'],
        #                                        x_box_min=self.opts['x_box_min'],
        #                                        x_box_max=self.opts['x_box_max'], alpha=self.opts['alpha'])
        #     elif Adversary in ['FGSM', 'PGD','DeepFool']:
        #         _, y_pred = torch.max(target_model(x), 1)
        #         "cal adv_x given different mode wf/shs. the mode of adversary set before"
        #         adv_x = self.x_adv_gen(x, y_pred, target_model, adversary)
        #
        #     else:
        #         print('No Adversary found! System will exit.')
        #         sys.exit()
        #
        #
        #     if self.mode == 'shs':
        #         adv_x = (adv_x.data.cpu().numpy().squeeze()*1500).round()
        #     elif self.mode == 'wf':
        #
        #         "if the data use L2 normalized, then need to inverse it back"
        #         # normalization = utils_wf.normalizer(x)
        #         # adv_x = normalization.inverse_Normalizer(adv_x)
        #
        #         adv_x = adv_x.data.cpu().numpy().squeeze()
        #         adv_x = adv_x.squeeze()
        #     else:
        #         print('mode should in ["wf","shs"], system will exit.')
        #         sys.exit()
        #
        #     adv_xs.append(adv_x)
        #     labels.append(y.data.cpu().numpy().squeeze())
        #
        # adv_xs = pd.DataFrame(adv_xs)
        # adv_xs['label'] = labels
        # output_path = self.data_path + '/adv_train_' + self.opts['Adversary'] + '.csv'
        # adv_xs.to_csv(output_path, index=0)
        # print('adversary examples of train data is generated')


def main(opts):

    gen_advX = gen_adv_x(opts)
    gen_advX.generate()




def get_opts_wf(mode,Adversary,classifier_type):
    return {
        'train_data_path': '../data/wf/train_NoDef_burst.csv',
        'test_data_path': '../data/wf/test_NoDef_burst.csv',
        'mode': mode,
        'classifier_type':classifier_type,
        'alpha':10,
        'num_class': 95,
        'input_size': 512,
        'Adversary': Adversary,
        'batch_size': 1,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,

    }


def get_opts_wf_sg(mode,Adversary,classifier_type,single_gen):
    "sg: (single-input generation) simply generated adv examples for one input"
    return {
        "single_gen": single_gen,
        'input_data_path': '../data/NoDef/data_NoDef_burst.csv',
        'mode': mode,
        'classifier_type':classifier_type,
        'alpha':10,
        'num_class': 95,
        'input_size': 512,
        'Adversary': Adversary,
        'batch_size': 1,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,

    }



def get_opts_wf_kf(mode,Adversary,classifier_type,id):
    "K-Fold generation"
    return {
        'id': id,
        'train_data_path': '../data/wf/cross_val/traffic_train_%d.csv' % id,
        'test_data_path': '../data/wf/cross_val/traffic_test_%d.csv' % id,
        'mode': mode,
        'classifier_type':classifier_type,
        'alpha':10,
        'num_class': 95,
        'input_size': 512,
        'Adversary': Adversary,
        'batch_size': 1,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,

    }



def get_opts_wf_ow(mode,Adversary,classifier_type):
    return {
        'train_data_path': '../data/wf_ow/train_NoDef_mix.csv',
        'test_data_path': '../data/wf_ow/test_NoDef_Mon.csv',
        'test_data_path_UnMon': '../data/wf_ow/test_NoDef_UnMon.csv',
        'mode': mode,
        'classifier_type':classifier_type,
        'alpha':10,
        'num_class': 96,
        'input_size': 512,
        'Adversary': Adversary,
        'batch_size': 1,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,

    }


def get_opts_shs(mode,Adversary,classifier_type):
    return {
        'train_data_path': '../data/shs/traffic_train.csv',
        'test_data_path': '../data/shs/traffic_test.csv',
        'mode': mode,
        'classifier_type': classifier_type,
        'alpha':None,
        'num_class': 101,
        'input_size': 256,
        'Adversary': Adversary,
        'batch_size': 1,
        'pert_box':0.3,
        'x_box_min':-1,
        'x_box_max':0,

    }



if __name__ == '__main__':

    adveraries = ['GAN']                #['FGSM','DeepFool','PGD','GAN']
    mode = 'wf_kf'                     # ['wf','shs','wf_ow','wf_kf','wf_sg']
    classifier_type = 'cnn'         # ['lstm','cnn']
    k = 5       # K-Fold


    for adv in adveraries:

        if mode == 'wf':
            opts = get_opts_wf(mode,adv,classifier_type)
        elif mode == 'wf_ow':
            opts = get_opts_wf_ow(mode,adv,classifier_type)
        elif mode == 'shs':
            opts = get_opts_shs(mode,adv,classifier_type)
        elif  mode == 'wf_sg':
            opts = get_opts_wf_sg(mode,adv,classifier_type,single_gen=True)
        elif mode == 'wf_kf':
            for id in range(k):
                print('-'*30, '\n','%d fold....' % id)
                opts = get_opts_wf_kf(mode,adv,classifier_type,id)
                main(opts)

        if mode != 'wf_kf':
            main(opts)




