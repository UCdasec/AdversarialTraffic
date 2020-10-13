"""
mainly for large data in DL.
split data in KFold manner in a straitified way (same samples of each class,balanced),
and save the corresponding (train_i,test_i) data,
ready for perform dl tasks
"""

import os
os.sys.path.append('..')

from sklearn.model_selection import StratifiedKFold
from train import utils_wf


def get_split_id(X,y,k=5):
    "obtain split index, split based on label and into balance data"
    split_id = []
    skf = StratifiedKFold(n_splits=k)
    for train_id,test_id in skf.split(X,y):
        split_id.append((list(train_id),list(test_id)))

    return split_id


def get_output_name(type,id,Adversary):

    # path = '../data/wf/cross_val/'
    path = '../data/wf_kf/lstm/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    output = path + 'adv_%s_%s_%d.csv' % (type,Adversary,id)

    return output


def gen_split_file(split_id,X,y,Adversary):
    """
    :param split_id: index for splitting
    :param X: X data need to be splitted
    :param y: y label to split
    :return:
    """
    "save splitted file in csv"
    for i,id in enumerate(split_id):
        print('*'*30)
        print('split %d' % i)
        train_id = id[0]
        test_id = id[1]
        print('train_id',len(train_id))
        print('test id',len(test_id))
        # for train_id, test_id in id:
        df_train = utils_wf.convert2dataframe(X.iloc[train_id],y.iloc[train_id])
        df_test = utils_wf.convert2dataframe(X.iloc[test_id],y.iloc[test_id])
        utils_wf.write2csv(df_train,get_output_name('train',i,Adversary))
        utils_wf.write2csv(df_test,get_output_name('test',i,Adversary))


if __name__ == '__main__':

    # set k-fold
    k = 5

    #load data before splitting
    # path = '../data/NoDef/data_NoDef_burst.csv'

    Adversary = ['FGSM','DeepFool','PGD','GAN']
    for adv in Adversary:
        path = '../data/wf_kf/lstm/adv_data/adv_%s.csv' % adv
        X,y = utils_wf.load_csv_data(path)

        #obtain split index, which is stratified
        split_id = get_split_id(X,y,k)

        # get and save the splitted data
        gen_split_file(split_id,X,y,adv)









