import os
os.sys.path.append('..')

import torch
import torch.nn.functional as F
from train import models
from train import utils_wf,utils_shs
import os,sys, nni, logging



LOG = logging.getLogger('lstm_param')




class train_lstm:
    def __init__(self,opts,params):

        "default params for target model"
        self.params = params

        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode + '/lstm'
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABEL: ', torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def train_model(self):

        if self.mode == 'wf':
            "load data"
            train_data = utils_wf.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            # params = utils_wf.params_lstm(self.opts['num_class'],self.opts['input_size'],self.opts['batch_size'])
            target_model = models.lstm(self.params).to(self.device)
            target_model.train()

        elif self.mode == 'shs':
            "load data"
            train_data = utils_shs.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            # params = utils_shs.params_lstm(self.opts['num_class'],self.opts['input_size'],self.opts['batch_size'])
            target_model = models.lstm(self.params).to(self.device)
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

                    "report intermediate result"
                    nni.report_intermediate_result(accuracy)


            "save model every 10 epochs"
            if epoch % 10 == 0:
                targeted_model_path = self.model_path + '/target_model.pth'
                torch.save(target_model.state_dict(), targeted_model_path)


        "test target model"
        target_model.eval()

        num_correct = 0
        total_instances = 0
        for i, data in enumerate(test_data, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            pred_lab = torch.argmax(target_model(test_x), 1)
            num_correct += torch.sum(pred_lab == test_y, 0)
            total_instances += len(test_y)

        acc = num_correct.item() / total_instances
        print('accuracy of target model against test dataset: %f\n' % (acc))

        "report final result "
        nni.report_final_result(acc)




def main(opts,params):
    trainTargetModel = train_lstm(opts,params)
    trainTargetModel.train_model()


def get_opts_wf(mode):
    return{
        'mode':mode,
        'num_class': 95,
        'input_size': 512,
        'batch_size': 64,
        'epochs':50,
        'lr': 0.001,
        'train_data_path': '../../data/wf/train_NoDef_burst.csv',
        'test_data_path': '../../data/wf/test_NoDef_burst.csv',
    }

def get_opts_shs(mode):
    return{
        'mode':mode,
        'num_class': 101,
        'input_size': 256,
        'batch_size': 64,
        'epochs':50,
        'lr':0.001,
        'train_data_path': '../data/shs/traffic_train.csv',
        'test_data_path': '../data/shs/traffic_test.csv',
    }


# if __name__ == '__main__':
#
#     mode = 'wf'
#
#     if mode == 'wf':
#         opts = get_opts_wf(mode)
#         params = utils_wf.params_lstm(opts['num_class'], opts['input_size'], opts['batch_size'])
#     elif mode == 'shs':
#         opts = get_opts_shs(mode)
#         params = utils_shs.params_lstm(opts['num_class'], opts['input_size'], opts['batch_size'])
#
#     main(opts,params)



if __name__ == '__main__':

    mode = 'wf'

    if mode == 'wf':
        opts = get_opts_wf(mode)
        params = utils_wf.params_lstm(opts['num_class'], opts['input_size'], opts['batch_size'])
    elif mode == 'shs':
        opts = get_opts_shs(mode)
        params = utils_shs.params_lstm(opts['num_class'], opts['input_size'], opts['batch_size'])

    try:
        "get param from tuner"
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = params
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)

        main(opts,PARAMS)

    except Exception as exception:
        LOG.exception(exception)
        raise

