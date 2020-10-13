import os
os.sys.path.append('..')

import torch
import torch.nn.functional as F
from train import models
from train import utils_wf,utils_shs
import os,sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from tools.detect_adv_ml import heat_map



class detect_adversary:
    def __init__(self,opts):

        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode + '/' + opts['classifier']
        # self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = ('cpu')
        print('CUDA AVAILABEL: ', torch.cuda.is_available())
        print('model path', self.model_path)

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')
        elif self.mode == 'detect':
            print('detecting adversary...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def train_model(self):

        if self.mode == 'wf' or 'detect':
            "load data"
            if mode == 'wf':
                train_data = utils_wf.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
                # test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])
            elif mode == 'detect':
                train_data = utils_wf.load_NormData_main(self.opts['train_data_path'], self.opts['batch_size'])
                # test_data = utils_wf.load_NormData_main(self.opts['test_data_path'], self.opts['batch_size'])

            "load target model structure"
            if self.opts['classifier'] == 'lstm':
                params = utils_wf.params_lstm_detect(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)
            elif self.opts['classifier'] == 'rnn':
                params = utils_wf.params_rnn(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.rnn(params).to(self.device)
            elif self.opts['classifier'] == 'cnn':
                params = utils_wf.params_cnn_detect(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn(params).to(self.device)
            elif self.opts['classifier'] == 'fcnn':
                params = utils_wf.params_fcnn(self.opts['num_class'],self.opts['input_size'])
                target_model = models.fcnn(params).to(self.device)
            target_model.train()

        elif self.mode == 'shs':
            "load data"
            train_data = utils_shs.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params_lstm(self.opts['num_class'],self.opts['input_size'],self.opts['batch_size'])
            target_model = models.lstm(params).to(self.device)
            target_model.train()

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        "train process"
        optimizer = torch.optim.Adam(target_model.parameters(),lr=self.opts['lr'])

        for epoch in range(self.opts['epochs']):
            loss_epoch = 0
            for i, data in enumerate(train_data, 0):
                train_x, train_y = data
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                "batch_first = False"
                if not self.opts['batch_size']:
                    train_x = train_x.transpose(0,1)

                optimizer.zero_grad()
                logits_model = target_model(train_x)
                loss_model = F.cross_entropy(logits_model, train_y)
                loss_epoch += loss_model

                loss_model.backward(retain_graph=True)
                optimizer.step()

                if i % 100 == 0:
                    _, predicted = torch.max(logits_model, 1)
                    correct = int(sum(predicted == train_y))
                    accuracy = correct / len(train_y)
                    msg = 'Epoch {:5}, Step {:5}, Loss: {:6.2f}, Accuracy:{:8.2%}.'
                    print(msg.format(epoch, i, loss_model, accuracy))

                "empty cache"
                del train_x,train_y
                torch.cuda.empty_cache()

            "save model every 10 epochs"
            if epoch % 10 == 0:
                targeted_model_path = self.model_path + '/target_model.pth'
                torch.save(target_model.state_dict(), targeted_model_path)


        "test target model"
        # target_model.eval()
        #
        # num_correct = 0
        # total_instances = 0
        # y_test = []
        # y_pred = []
        # for i, data in enumerate(test_data, 0):
        #     test_x, test_y = data
        #     test_x, test_y = test_x.to(self.device), test_y.to(self.device)
        #     pred_lab = torch.argmax(target_model(test_x), 1)
        #     num_correct += torch.sum(pred_lab == test_y, 0)
        #     total_instances += len(test_y)
        #
        #     "save result"
        #     y_test += (test_y.cpu().numpy().tolist())
        #     y_pred += (pred_lab.cpu().numpy().tolist())
        #
        # print('{} with {}'.format(self.opts['mode'], self.opts['classifier']))
        #
        # print(classification_report(y_test, y_pred))
        # print('confusion matrix is {}'.format(confusion_matrix(y_test, y_pred)))
        # print('accuracy of target model against test dataset: %f\n' % (num_correct.item() / total_instances))
        # print('accuracy is {}'.format(metrics.accuracy_score(y_test, y_pred)))
        #
        # # plot confusion matrix
        # cm = confusion_matrix(y_test, y_pred)
        # heat_map(cm, self.opts['classifier'], namorlize=True)


    #----------------------------------
    def test_model(self):

        "load data and target model"
        if self.mode == 'wf' or 'detect':
            "load data"
            if mode == 'wf':
                # train_data = utils_wf.load_data_main(self.opts['train_data_path'], self.opts['batch_size'])
                test_data = utils_wf.load_data_main(self.opts['test_data_path'], self.opts['batch_size'])
            elif mode == 'detect':
                # train_data = utils_wf.load_NormData_main(self.opts['train_data_path'], self.opts['batch_size'])
                test_data = utils_wf.load_NormData_main(self.opts['test_data_path'], self.opts['batch_size'])

            "load target model structure"
            if self.opts['classifier'] == 'lstm':
                params = utils_wf.params_lstm_detect(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)
            elif self.opts['classifier'] == 'rnn':
                params = utils_wf.params_rnn(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.rnn(params).to(self.device)
            elif self.opts['classifier'] == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.cnn(params).to(self.device)
            elif self.opts['classifier'] == 'fcnn':
                params = utils_wf.params_fcnn(self.opts['num_class'], self.opts['input_size'])
                target_model = models.fcnn(params).to(self.device)
            target_model.eval()

        elif self.mode == 'shs':
            "load data"
            # train_data = utils_shs.load_data_main(self.opts['train_data_path'], self.opts['batch_size'])
            test_data = utils_shs.load_data_main(self.opts['test_data_path'], self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params_lstm(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
            target_model = models.lstm(params).to(self.device)
            target_model.train()

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        model_name = self.model_path + '/target_model.pth'

        model_weights = torch.load(model_name,map_location=self.device)
        for k in model_weights: print(k)
        # for k in model_weights['shared_layers']: print("Shared layer", k)


        target_model.load_state_dict(torch.load(model_name, map_location=self.device))
        target_model.eval()

        num_correct = 0
        total_instances = 0
        y_test = []
        y_pred = []
        for i, data in enumerate(test_data, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            pred_lab = torch.argmax(target_model(test_x), 1)
            num_correct += torch.sum(pred_lab == test_y, 0)
            total_instances += len(test_y)

            "save result"
            y_test += (test_y.cpu().numpy().tolist())
            y_pred += (pred_lab.cpu().numpy().tolist())

        print('{} with {}'.format(self.opts['mode'], self.opts['classifier']))

        print(classification_report(y_test, y_pred))
        print('confusion matrix is {}'.format(confusion_matrix(y_test, y_pred)))
        print('accuracy of target model against test dataset: %f\n' % (num_correct.item() / total_instances))
        print('accuracy is {}'.format(metrics.accuracy_score(y_test, y_pred)))

        # plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        heat_map(cm, self.opts['classifier'],namorlize=True)



def main(opts):
    trainDetector = detect_adversary(opts)
    trainDetector.train_model()


def main_test(opts):
    trainDetector = detect_adversary(opts)
    trainDetector.test_model()


def get_opts_wf(mode,classifier):
    return{
        'mode':mode,
        'classifier': classifier,
        'num_class': 95,
        'input_size': 512,
        'batch_size': 64,
        'epochs':50,
        'lr': 0.006,
        'train_data_path': '../data/wf/train_NoDef_burst.csv',
        'test_data_path': '../data/wf/test_NoDef_burst.csv',
    }

def get_opts_shs(mode,classifier):
    return{
        'mode':mode,
        'classifier': classifier,
        'num_class': 101,
        'input_size': 256,
        'batch_size': 64,
        'epochs':50,
        'lr':0.001,
        'train_data_path': '../data/shs/traffic_train.csv',
        'test_data_path': '../data/shs/traffic_test.csv',
    }



def get_opts_detect(mode,classifier):
    "detect adversary"
    return {
        'mode': mode,
        'classifier': classifier,
        'num_class': 4,
        'input_size': 512,
        'batch_size': 16,
        'epochs': 50,
        'lr': 0.01,
        'train_data_path': '../data/wf/cnn/adv_train_all.csv',
        'test_data_path': '../data/wf/cnn/adv_test_all.csv',
    }


if __name__ == '__main__':

    mode = 'detect'                     # ['wf','shs','detect']
    classifier = 'cnn'                  # ['cnn','rnn','lstm']

    if mode == 'wf':
        opts = get_opts_wf(mode)
    elif mode == 'shs':
        opts = get_opts_shs(mode)
    elif mode == 'detect':
        opts = get_opts_detect(mode,classifier)

    #tain mode
    main(opts)

    #test mode
    # main_test(opts)
