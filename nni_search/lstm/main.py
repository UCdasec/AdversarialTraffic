import os
os.sys.path.append('..')

from nni_search.lstm import train_lstm
from train import utils_wf,utils_shs
import nni,logging
LOG = logging.getLogger('lstm_param')





def get_opts_wf(mode):
    return{
        'mode':mode,
        'num_class': 95,
        'input_size': 512,
        'batch_size': 64,
        'epochs':1,
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
        'epochs':2,
        'lr':0.001,
        'train_data_path': '../data/shs/traffic_train.csv',
        'test_data_path': '../data/shs/traffic_test.csv',
    }



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
        model = train_lstm.train_lstm(opts,PARAMS)
        model.train_model()
    except Exception as exception:
        LOG.exception(exception)
        raise

