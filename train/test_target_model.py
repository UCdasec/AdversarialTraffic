"test the performance of target model"

import os
os.sys.path.append('..')

from train import models,utils_wf,utils_shs
import torch
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics


class performance_target_model:
    def __init__(self,opts):
        self.opts = opts
        self.mode = opts['mode']
        self.classifier_type = opts['classifier_type']
        self.model_path = '../model/'  + self.mode + '/' + opts['classifier_type']
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABEL:', torch.cuda.is_available())
        print('testing model path:  ', self.model_path)


    def test_peformance(self):

        "load data and target model"
        if self.mode == 'wf' or 'detect' or 'wf_ow' or 'wf_tf':
            "load data"
            test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'],self.opts['input_size'])
                target_model = models.cnn_norm(params).to(self.device)

            elif self.classifier_type == 'lstm':
                if self.mode == 'wf_ow':
                    params = utils_wf.params_lstm_ow(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                elif self.mode == 'wf_kf':
                    params = utils_wf.params_lstm(self.opts['num_class'], self.opts['input_size'],self.opts['batch_size'])
                else:
                    params = utils_wf.params_lstm(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
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

        if self.mode == 'wf_kf':
            model_name = '/target_model_%d.pth' % self.opts['id']
        else:
            model_name = '/target_model.pth'
        model_path = self.model_path + model_name
        print(model_path)
        target_model.load_state_dict(torch.load(model_path, map_location=self.device))
        target_model.eval()


        "testing process..."

        num_correct = 0
        total_case = 0
        y_test = []
        y_pred = []
        for i, data in enumerate(test_data, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            "add softmax after fc of model to normalize output as positive values and sum =1"
            pred_lab = torch.argmax(torch.softmax(target_model(test_x),1), 1)
            # pred_lab = torch.argmax(target_model(test_x), 1)

            num_correct += torch.sum(pred_lab == test_y, 0)
            total_case += len(test_y)

            "save result"
            y_test += (test_y.cpu().numpy().tolist())
            y_pred += (pred_lab.cpu().numpy().tolist())

        # print('accuracy in testing set: %f\n' % (num_correct.item() / total_case))
        # print(classification_report(y_test, y_pred))
        print('confusion matrix is {}'.format(confusion_matrix(y_test, y_pred)))
        print('accuracy is {}'.format(metrics.accuracy_score(y_test, y_pred)))


def main(opts):

    test_target_model = performance_target_model(opts)
    test_target_model.test_peformance()


def get_opts_wf(mode, classifier_type):

    return{
        'mode':mode,
        'classifier_type':classifier_type,
        'batch_size':64,
        'num_class':95,
        'input_size':512,
        'test_data_path': '../data/wf/test_NoDef_burst.csv'
    }


def get_opts_wf_kFold(mode,classifier_type,id):
    "website fingerprinting with 5-fold cross validation"
    return{
        'id': id,
        'mode':mode,
        'classifier_type': classifier_type,
        'num_class': 95,
        'input_size': 512,
        'batch_size': 64,
        'test_data_path': '../data/wf/cross_val/traffic_test_%d.csv' % id,
    }



def get_opts_wf_ow(mode, classifier_type):

    return{
        'mode':mode,
        'classifier_type':classifier_type,
        'batch_size':64,
        'num_class':96,
        'input_size':512,
        'test_data_path': '../data/wf_ow/test_NoDef_Mon.csv'
    }


def get_opts_shs(mode, classifier_type):
    return {
        'mode': mode,
        'classifier_type': classifier_type,
        'batch_size': 64,
        'num_class': 101,
        'input_size': 256,
        'test_data_path': '../data/shs/traffic_test.csv'
    }




def get_opts_detect(mode,classifier):
    "detect adversary"
    return {
        'mode': mode,
        'classifier_type': classifier,
        'num_class': 4,
        'input_size': 512,
        'batch_size': 64,
        'epochs': 50,
        'lr': 0.006,
        'train_data_path': '../data/wf/cnn/adv_train_all.csv',
        'test_data_path': '../data/wf/cnn/adv_test_all.csv',
    }


if __name__ == '__main__':
    mode = 'wf_kf'                 # ['wf','wf_ow','shs','detect','wf_tf']
    classifier_type = 'lstm'         # ['cnn','lstm','rnn']

    if mode == 'wf':
        opts = get_opts_wf(mode, classifier_type)
    elif mode == 'wf_ow':
        opts = get_opts_wf_ow(mode, classifier_type)
    elif mode == 'shs':
        opts = get_opts_shs(mode, classifier_type)
    elif mode == 'detect':
        opts = get_opts_detect(mode,classifier_type)
    elif mode == 'wf_kf':
        k = 5   # num of K-Fold
        for id in range(k):
            opts = get_opts_wf_kFold(mode,classifier_type,id)
            main(opts)
        print('K-Fold function completed, system is out')
        sys.exit()
    else:
        print('mode should in ["wf","shs","detect"]. system will exit.')
        sys.exit()

    main(opts)