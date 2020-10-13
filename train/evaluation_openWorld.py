"""
test the performance in the open world settings.
the way to calculate TP FP TN FN in open world setting is based on
https://github.com/deep-fingerprinting/df/blob/master/src/OpenWorld_DF_NoDef_Evaluation.py
"""


import os
os.sys.path.append('..')

from train import models,utils_wf
import torch
import sys
import numpy as np


class performance_target_model:
    def __init__(self,opts):
        self.opts = opts
        self.mode = opts['mode']
        self.classifier_type = opts['classifier_type']
        self.model_type = opts['model_type']
        self.model_path = '../model/'  + self.mode + '/' + opts['classifier_type']
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABEL:', torch.cuda.is_available())
        print('testing model path:  ', self.model_path)

        "create log file to record results"
        results_output = '../result/%s_%s_%s.csv' % (self.mode, self.classifier_type,self.opts['Adversary'])
        self.log_file = open(results_output, 'w')
        self.log_file.writelines("%s,%s,%s,%s,%s,%s  ,%s  ,  %s, %s\n" % (
        'Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'Precision', 'Recall'))



    def test_peformance(self,threshold_val):

        "load data and target model"
        if self.mode == 'wf_ow':
            "load data"
            if self.opts['test_data_type'] == 'NoDef':
                test_data_Mon = utils_wf.load_data_main(self.opts['test_data_Mon_path'],self.opts['batch_size'])
                test_data_UnMon = utils_wf.load_data_main(self.opts['test_data_UnMon_path'],self.opts['batch_size'])
            elif self.opts['test_data_type'] == 'Def':
                test_data_Mon = utils_wf.load_data_main(self.opts['adv_test_data_Mon_path'],self.opts['batch_size'])
                test_data_UnMon = utils_wf.load_data_main(self.opts['adv_test_data_UnMon_path'],self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'],self.opts['input_size'])
                target_model = models.cnn_norm(params).to(self.device)

            elif self.classifier_type == 'lstm':
                params = utils_wf.params_lstm_ow_eval(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)

        else:
            print('mode not in ["wf_ow"], system will exit.')
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

        print('target model %s' % model_name)
        print('test data path unmonitored %s ' % self.opts['adv_test_data_UnMon_path'])
        print('test data path monitored %s ' % self.opts['adv_test_data_Mon_path'])

        # ==============================================================
        "testing process..."
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        "obtain monitored and Unmonitored lables"
        _,y_test_Mon = utils_wf.load_csv_data(self.opts['test_data_Mon_path'])
        _,y_test_UnMon = utils_wf.load_csv_data(self.opts['test_data_UnMon_path'])
        monitored_labels = torch.Tensor(y_test_Mon).long().to(self.device)
        unmonitored_labels = torch.Tensor(y_test_UnMon).long().to(self.device)

        # ==============================================================
        # Test with Monitored testing instances

        for i, data in enumerate(test_data_Mon, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            max_probs, pred_labels = torch.max(torch.softmax(target_model(test_x),1), 1)

            for j, pred_label in enumerate(pred_labels):
                if pred_label in monitored_labels:  # predited as Monitored
                    if max_probs[j] >= threshold_val:   # probability greater than the threshold
                        TP += 1
                    else: # predicted as unmonitored and true lable is Monitored
                        FN += 1
                elif pred_label in unmonitored_labels: # predicted as unmonitored and true lable is monitored
                    FN += 1

        # ==============================================================
        # Test with Unmonitored testing instances
        for i, data in enumerate(test_data_UnMon, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            max_probs, pred_labels = torch.max(torch.softmax(target_model(test_x),1), 1)

            for j, pred_label in enumerate(pred_labels):
                if pred_label in unmonitored_labels:  # predited as unmonitored and true label is unmonitored
                    TN += 1
                elif pred_label in monitored_labels: # predicted as Monitored
                    if max_probs[j] >= threshold_val: # predicted as monitored and true label is unmonitored
                        FP += 1
                    else:
                        TN += 1

        "print result"
        print("TP : ", TP)
        print("FP : ", FP)
        print("TN : ", TN)
        print("FN : ", FN)
        print("Total  : ", TP + FP + TN + FN)
        TPR = float(TP) / (TP + FN)
        print("TPR : ", TPR)
        FPR = float(FP) / (FP + TN)
        print("FPR : ", FPR)
        Precision = float(TP) / (TP + FP)
        print("Precision : ", Precision)
        Recall = float(TP) / (TP + FN)
        print("Recall : ", Recall)
        print("\n")

        self.log_file.writelines("%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n" % (threshold_val, TP, FP, TN, FN, TPR, FPR, Precision, Recall))




def main(opts):

    test_target_model = performance_target_model(opts)

    thresholds = 1.0 - 1 / np.logspace(0.05, 2, num=15, base=10, endpoint=True)

    for threshold in thresholds:
        print('adversary training with %s against %s' %(opts['Adversary'],opts['Adversary']))

        print('threshold: ', threshold)
        test_target_model.test_peformance(threshold)


def get_opts_wf_ow(mode, Adversary,classifier_type,model_type,test_data_type):

    return{
        'mode':mode,
        'Adversary': Adversary,
        'classifier_type':classifier_type,
        'model_type' : model_type,
        'test_data_type': test_data_type,
        'batch_size':64,
        'num_class':96,
        'input_size':512,
        'test_data_Mon_path': '../data/wf_ow/test_NoDef_Mon.csv',
        'test_data_UnMon_path': '../data/wf_ow/test_NoDef_UnMon.csv',
        'adv_test_data_Mon_path': '../data/wf_ow/%s/adv_test_Mon_%s.csv' % (classifier_type,Adversary),
        'adv_test_data_UnMon_path': '../data/wf_ow/%s/adv_test_UnMon_%s.csv' % (classifier_type,Adversary),

    }




if __name__ == '__main__':
    mode = 'wf_ow'                 # ['wf_ow','wf','shs','detect']
    Adversary = ['GAN']  # 'FGSM', 'DeepFool', 'PGD', 'GAN'
    classifier_type = 'cnn'  # ['cnn','lstm','rnn']

    "two combinations [adv_target_model,Def], [target_model,NoDef] at this point"
    model_type = 'adv_target_model'     #[adv_target_model,target_model]
    test_data_type = 'Def'        #[NoDef, Def], NoDef: original test data without adding defense, Def: defended data with defense included



    if test_data_type == 'Def':
        for adv in Adversary:
            print('adversary is: ',adv)
            opts = get_opts_wf_ow(mode,adv,classifier_type,model_type,test_data_type)
            main(opts)

    else:
        adv = '0'
        opts = get_opts_wf_ow(mode, adv, classifier_type, model_type, test_data_type)
        main(opts)
