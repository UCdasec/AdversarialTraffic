import os
os.sys.path.append('..')

import torch
from train import models,utils_wf,utils_shs
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import copy,sys



class adv_train_deepfool:

    def __init__(self,opts):

        self.opts = opts
        self.mode = opts['mode']
        self.classifier_type = opts['classifier_type']
        self.model_path = '../model/' + self.mode + '/' + self.classifier_type
        self.data_path = '../data/' + self.mode + '/'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABLE:', torch.cuda.is_available())
        print('Mode: ', self.mode)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        print('model path:  ',self.model_path)


    # def x_adv_gen(self,x, y, model, adversary):
    #     """
    #     for white-box based adv training, target model training should separate with adv examples generation.
    #     cause adv_training related to target model, and adv example generation involved target model.
    #     therefore, should set target model as evaluation mode when producing adv examples in adv training
    #     """
    #     model_cp = copy.deepcopy(model)
    #     for p in model_cp.parameters():
    #         p.requires_grad = False
    #     model_cp.eval()
    #
    #     adversary.model = model_cp
    #     _,x_adv = adversary.perturbation(x, y,self.opts['alpha'])
    #
    #     return x_adv


    def adv_train_process(self,delay=0.5):
        "Adversarial training, returns pertubed mini batch"
        "delay: parameter to decide how many epochs should be used as adv training"

        if self.opts['Adversary'] == 'WT':
            adv_train_data_path = '../data/WalkieTalkie/defended_csv/adv_train_WT.csv'
        else:
            adv_train_data_path = self.data_path + self.classifier_type + '/' + self.opts['adv_train_data_path']
        print('adv train data path: ',adv_train_data_path)

        if self.mode in ['wf','wf_ow','wf_kf']:
            "load data"
            if self.mode != 'wf_kf':
                train_path = self.data_path + self.opts['train_data_path']
                train_data = utils_wf.load_data_main(train_path,self.opts['batch_size'],shuffle=True)
                # test_data = utils_wf.load_data_main(self.data_path + self.opts['test_data_path'],self.opts['batch_size'])
            else:
                # benign_path = '../data/wf/cross_val/'
                # train_path =  benign_path + self.opts['train_data_path']
                train_path = '../data/wf/train_NoDef_burst.csv'
                train_data = utils_wf.load_data_main(train_path,self.opts['batch_size'], shuffle=True)
                # test_data = utils_wf.load_data_main(benign_path + self.opts['test_data_path'],self.opts['batch_size'])
            print('train data path: ',train_path)

            adv_train_data = utils_wf.load_data_main(adv_train_data_path,self.opts['batch_size'],shuffle=True)

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn_norm(params).to(self.device)
                target_model.train()

            elif self.classifier_type == 'lstm':
                if self.mode == 'wf_ow':
                    params = utils_wf.params_lstm_ow(self.opts['num_class'], self.opts['input_size'],self.opts['batch_size'])
                else:
                    params = utils_wf.params_lstm(self.opts['num_class'], self.opts['input_size'],self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)
                target_model.train()


        elif self.mode == 'shs':
            "load data"
            train_data = utils_shs.load_data_main(self.data_path + self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_shs.load_data_main(self.data_path + self.opts['test_data_path'],self.opts['batch_size'])
            adv_train_data = utils_shs.load_data_main(adv_train_data_path, self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_shs.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn_noNorm(params).to(self.device)
                target_model.train()

            elif self.classifier_type == 'lstm':
                params = utils_shs.params_lstm(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)
                target_model.train()

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(target_model.parameters(), lr=1e-3)


        "start training"
        steps = 0
        flag = False
        start_time = datetime.now()
        # print('adversary working dir {}'.format(adv_train_data_path))
        for epoch in range(self.opts['epochs']):
            print('Starting epoch %d / %d' % (epoch + 1, self.opts['epochs']))

            if flag:
                print('{} based adversarial training...'.format(self.opts['Adversary']))

            for (x,y),(x_adv,y_adv) in zip(train_data,adv_train_data):
                steps += 1
                optimizer.zero_grad()

                x,y = x.to(self.device), y.to(self.device)
                outputs = target_model(x)
                loss = loss_criterion(outputs,y)

                "adversarial training"
                if epoch + 1 >= int((1-delay)*self.opts['epochs']):
                    x_adv, y_adv = x_adv.to(self.device), y_adv.to(self.device)
                    flag = True
                    loss_adv = loss_criterion(target_model(x_adv),y_adv)
                    loss = (loss + loss_adv) / 2

                "print results every 100 steps"
                if steps % 100 == 0:

                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Time usage:{:9}."
                    print(msg.format(steps, loss, time_usage))

                loss.backward()
                optimizer.step()

            if epoch!=0 and epoch % 10 == 0 or epoch == self.opts['epochs']:
                if self.mode != 'wf_kf':
                    output_name = '/adv_target_model_%s.pth' % self.opts['Adversary']
                else:
                    output_name = '/adv_target_model_%s_%d.pth' % (self.opts['Adversary'], self.opts['id'])
                output_path = self.model_path + output_name
                torch.save(target_model.state_dict(), output_path)


        "test trianed model"
        # target_model.eval()
        #
        #
        # test_loss = 0.
        # test_correct = 0
        # total_case = 0
        # start_time = datetime.now()
        # i = 0
        # for data, label in test_data:
        #     i += 1
        #     print('testing {}'.format(i))
        #     data, label = data.to(self.device), label.to(self.device)
        #     outputs = target_model(data)
        #     loss = loss_criterion(outputs, label)
        #     test_loss += loss * len(label)
        #     _, predicted = torch.max(outputs, 1)
        #     correct = int(sum(predicted == label))
        #     test_correct += correct
        #     total_case += len(label)
        #
        #     # delete caches
        #     del data, label, outputs, loss
        #     torch.cuda.empty_cache()
        #
        # accuracy = test_correct / total_case
        # loss = test_loss / total_case
        # print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))
        #
        # end_time = datetime.now()
        # time_diff = (end_time - start_time).seconds
        # print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))




    def testing_process(self):

        print('testing mode...')

        if self.mode == 'wf':
            "load data"
            test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.target_model_wf(params).to(self.device)

        elif self.mode == 'shs':
            "load data"
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.target_model_shs(params).to(self.device)

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        model_name = self.model_path + '/adv_target_model_' + self.opts['Adversary'] + '.pth'
        target_model.load_state_dict(torch.load(model_name, map_location=self.device))
        target_model.eval()
        loss_criterion = nn.CrossEntropyLoss()

        test_loss = 0.
        test_correct = 0
        total_case = 0
        start_time = datetime.now()
        i = 0
        for data, label in test_data:
            i += 1
            print('testing {}'.format(i))
            data, label = data.to(self.device),label.to(self.device)
            outputs = target_model(data)
            loss = loss_criterion(outputs, label)
            test_loss += loss * len(label)
            _, predicted = torch.max(outputs, 1)
            correct = int(sum(predicted == label))
            test_correct += correct
            total_case += len(label)

            # delete caches
            del data, label, outputs, loss
            torch.cuda.empty_cache()

        accuracy = test_correct / total_case
        loss = test_loss / total_case
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))



def train_main(opts):
    adv_training = adv_train_deepfool(opts)
    adv_training.adv_train_process(delay=0.2)



def test_main(opts):
    adv_training = adv_train_deepfool(opts)
    adv_training.testing_process()


def get_opts_wf(mode,Adversary,classifier_type):

    return {
        'mode':mode,
        'Adversary': Adversary,
        'classifier_type': classifier_type,
        'input_size':512,
        'num_class':95,
        'epochs': 50,
        'batch_size': 64,
        'train_data_path': 'train_NoDef_burst.csv',
        'test_data_path': 'test_NoDef_burst.csv',
        'adv_train_data_path': 'adv_train_' + Adversary +'.csv',

    }


def get_opts_wf_kf(mode,Adversary,classifier_type,id):

    return {
        'id':id,
        'mode':mode,
        'Adversary': Adversary,
        'classifier_type': classifier_type,
        'input_size':512,
        'num_class':95,
        'epochs': 50,
        'batch_size': 64,
        'train_data_path': 'traffic_train_%d.csv' % id,
        'test_data_path': 'traffic_test_%d.csv' % id,
        'adv_train_data_path': 'adv_train_%s_%d.csv' % (Adversary,id),

    }


def get_opts_wf_ow(mode,Adversary,classifier_type):

    return {
        'mode':mode,
        'Adversary': Adversary,
        'classifier_type': classifier_type,
        'input_size':512,
        'num_class':96,
        'epochs': 50,
        'batch_size': 64,
        'train_data_path': 'train_NoDef_mix.csv',
        'test_data_path': 'test_NoDef_UnMon.csv',
        'adv_train_data_path': 'adv_train_' + Adversary +'.csv',

    }



def get_opts_shs(mode,Adversary,classifier_type):

    return {
        'mode':mode,
        'Adversary': Adversary,
        'classifier_type': classifier_type,
        'input_size':256,
        'num_class':101,
        'batch_size':64,
        'epochs':50,
        'train_data_path': 'traffic_train.csv',
        'test_data_path': 'traffic_test.csv',
        'adv_train_data_path': 'adv_train_' + Adversary + '.csv',

    }


if __name__ == '__main__':

    modes = ['wf_kf']                                   # ['wf','shs','wf_ow','wf_kf']
    Adversary = ['GAN','FGSM','DeepFool','PGD']                             # ['FGSM','DeepFool','PGD','GAN',WT']
    classifier_type = 'lstm'                          # ['lstm','cnn']
    k = 5 # K-fold

    for adv in Adversary:
        print('Adversary: ', adv)
        for mode in modes:
            if mode == 'wf':
                opts = get_opts_wf(mode,adv,classifier_type)
            elif mode == 'wf_ow':
                opts = get_opts_wf_ow(mode, adv, classifier_type)
            elif mode == 'shs':
                opts = get_opts_shs(mode,adv,classifier_type)
            elif mode == 'wf_kf':
                for id in range(k):
                    print('-' * 30, '\n', '%d fold....' % id)
                    opts = get_opts_wf_kf(mode, adv, classifier_type, id)
                    train_main(opts)

            if mode != 'wf_kf':
                train_main(opts)
