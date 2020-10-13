import os
os.sys.path.append('..')

from train import utils_wf

if __name__ == '__main__':
    path = '../data/wf_wang/wang_Mon.csv'
    data,_ = utils_wf.load_csv_data(path)
    dic = utils_wf.size_distribution(data,'wang_Mon')
    print(dic)