"""
preprocessing the pkl file data
and write it into csv file
"""

import os
os.sys.path.append('..')

from train import utils_wf
from sklearn.model_selection import train_test_split
import pandas as pd
import random


def pkl2burst_csv( slice_threshold):

    """
    load pkl file, transform data to burst and write it into csv
    No preprocessing: keep the original way. do not remove instance less than 50 packets and starting with incoming packet
    """

    type = 'valid'   #['test', 'train', 'valid']
    folder_name = 'wf_ow'       #[
    data_name = 'Nodef'   #['NoDef', 'WalkieTalkie' ]


    # x_path = '../data/' + folder_name + '/X_' + type + '_' + data_name + '.pkl'
    # out_path = '../data/' + folder_name + '/' + type + '_' + data_name + '.csv'
    # y_path = '../data/' + folder_name + '/y_' + type + '_' + data_name + '.pkl'

    # x_path = '../data/NoDef/X_valid_NoDef.pkl'
    # out_path = '../data/NoDef/valid_NoDef.csv'
    # y_path = '../data/NoDef/y_valid_NoDef.pkl'

    # x_path = '../data/NoDef/X_train_NoDef.pkl'
    # out_path = '../data/NoDef/train_NoDef.csv'
    # y_path = '../data/NoDef/y_train_NoDef.pkl'


    "processing pkl file"
    # x = utils_wf.load_pkl_data(x_path)
    # y = utils_wf.load_pkl_data(y_path)

    "processing wang data csv file"
    file_id = 'wang_UnMon'
    x_path = '../data/wf_wang/' + file_id +'.csv'
    out_path = '../data/wf_wang/' + file_id + '_burst.csv'
    x,y = utils_wf.load_csv_data(x_path)

    # X_new, y_new = utils_wf.data_preprocess(x, y) #remove trace less than 50 packets and starting with incoming packet
    x_burst,x_burst_noSlicing = utils_wf.burst_transform(x,slice_threshold)

    # utils_wf.size_distribution(x_burst_noSlicing,file_id)

    burst_data = utils_wf.convert2dataframe(x_burst,y)
    utils_wf.write2csv(burst_data,out_path)



def merge_data(slice_threshold):
    "merge train/test/valid data into one csv file processed source data (remove less than 50 packets and starting with incoming packet)"
    "output data in burst format"

    data_folder = '../data/wf_ow/'
    out_folder = data_folder + '/input_size_' + str(slice_threshold) + '/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    X_path = [data_folder + 'X_train_NoDef.pkl',data_folder + 'X_test_Unmon_NoDef.pkl',data_folder + 'X_test_Mon_NoDef.pkl',data_folder + 'X_valid_NoDef.pkl']
    Y_path = [data_folder + 'y_train_NoDef.pkl',data_folder + 'y_test_Unmon_NoDef.pkl',data_folder + 'y_test_Mon_NoDef.pkl',data_folder + 'y_valid_NoDef.pkl']
    out_path = out_folder+ 'data_NoDef_processed.csv'

    X,Y = [],[]
    for x_path,y_path in zip(X_path,Y_path):
        X += utils_wf.load_pkl_data(x_path)
        Y += utils_wf.load_pkl_data(y_path)
    print('data instances after merged: {}'.format(len(Y)))

    "remove less than 50 packets and starting with incoming packet"
    X_new,Y_new = utils_wf.data_preprocess(X, Y)
    print('data instances after processed: {}'.format(len(Y_new)))

    "convert to burst"
    x_burst, _ = utils_wf.burst_transform(X_new, slice_threshold)
    data_new = utils_wf.convert2dataframe(x_burst,Y_new)
    utils_wf.write2csv(data_new,out_path)


def get_ow_train_test(path,outpath_train,outpath_test):
    """
    extract opened world instances,  random take 20,000  training set
    the rest of ow instances as ow testing data
    output csv file respectively
    :return:
    """

    "get ow instnace"
    x,y = utils_wf.get_ow_data(path)
    x = pd.DataFrame(x)
    candit_list = list(range(0,len(y)))
    rand_list = random.sample(candit_list,20000)
    X_train,X_test,y_train,y_test = [],[],[],[]
    for i in range(len(y)):
        if i in rand_list:
            X_train.append(x.iloc[i, :])
            y_train.append(y[i])
        else:
            X_test.append(x.iloc[i,:])
            y_test.append(y[i])

    train_ow = utils_wf.convert2dataframe(X_train,y_train)
    test_ow = utils_wf.convert2dataframe(X_test,y_test)

    utils_wf.write2csv(train_ow, outpath_train)
    utils_wf.write2csv(test_ow, outpath_test)


def build_ow_train(path1,path2,outpath):
    """
    concat two dataset
    :param path1: ow train data
    :param path2: cw train data
    :return:
    """

    X_1,y_1 = utils_wf.load_csv_data(path1)
    X_2,y_2 = utils_wf.load_csv_data(path2)

    train_1 = utils_wf.convert2dataframe(X_1,y_1)
    train_2 = utils_wf.convert2dataframe(X_2,y_2)

    merge = [train_1,train_2]
    train_merge = pd.concat(merge)

    utils_wf.write2csv(train_merge,outpath)



def tranform2burst(path,out_path,slice_threshold):
    "load processed source data from csv, transform to burst with certain fixed size and write it in csv"


    x,y = utils_wf.load_csv_data(path)
    x_burst, x_burst_nopadding = utils_wf.burst_transform(x, slice_threshold)
    burst_data = utils_wf.convert2dataframe(x_burst,y,mode='padding')
    utils_wf.write2csv(burst_data,out_path)



def extract_balance_data(path,num,outpath,data_type):
    "extract num instances for each class, write into csv"

    if data_type == 'cw':    #[ow,cw]
        X,Y = utils_wf.extract_data_each_class(path,num)
    elif data_type == 'ow':
        X, Y = utils_wf.extract_data_each_class_ow(path, num)
    data = utils_wf.convert2dataframe(X,Y,mode='NoPadding')
    utils_wf.write2csv(data,outpath)




def test_train_split(path,out_train_path,out_test_path,test_size=0.2):
    "split processed data into train and test, write in csv"

    X,y = utils_wf.load_csv_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)

    X_test['label'] = y_test
    X_train['label'] = y_train
    X_test.to_csv(out_test_path, index=0)
    X_train.to_csv(out_train_path, index=0)

    print('No of instances of train data: {}'.format(len(X_train)))
    print('No of instances of test data: {}'.format(len(X_test)))


def normalize_data(path,out_path):
    "normlize data with MinMaxScaler to [-1,1], write it to csv"

    X,y = utils_wf.load_csv_data(path)

    X = utils_wf.data_normalize(X)
    X = pd.DataFrame(X)
    X['label'] = y
    X.to_csv(out_path,index=0)



def statistic_info(path):
    "the minimum number of instances for each class"

    _,Y = utils_wf.load_csv_data(path)
    statis = {}

    for y in Y:
        if y not in statis:
            statis[y] = 1
        else:
            statis[y] += 1

    values = statis.values()
    mini = min(values)
    print('the smallest number instances for each class among all class is: {}'.format(mini))



if __name__ == '__main__':

    #*************************************
    "source pkl to burst csv"
    slice_threshold = 512
    pkl2burst_csv(slice_threshold)

    #*************************************
    "merge train/test/valid NoDef data as processed source data"
    # slice_threshold = 1024
    # merge_data(slice_threshold)


    #*************************************
    "statistic information"
    # data_folder = '../data/wf_ow/'
    # path = data_folder + 'data_NoDef_processed.csv'
    # statistic_info(path)



    #*************************************
    """
    closed-world: extract balance data from source data, num instances per class
    opened-world: extract balance data for closed-data part, and include all ow data
    """

    # data_type = 'ow'  # ['ow','cw']
    # data_folder = '../data/wf_ow/'
    # path = data_folder + 'data_NoDef_processed.csv'
    # outpath = data_folder + 'data_NoDef_burst.csv'
    # num = 460       # num of instance per class
    # extract_balance_data(path,num,outpath,data_type)



    # *************************************
    """
    get ow instances 
    20000 ow data as training data
    rest as testing data 
    """

    # data_folder = '../data/wf_ow/input_size_1024/'
    # path = data_folder + 'data_NoDef_processed.csv'
    # outpath_train = data_folder + 'train_NoDef_UnMon.csv'
    # outpath_test = data_folder + 'test_NoDef_UnMon.csv'
    #
    # get_ow_train_test(path,outpath_train,outpath_test)


    # *************************************
    """
    build ow dataset  
    20000 ow data add to cw training data
    """

    # data_folder = '../data/wf_ow/input_size_1024/'
    # path_cw = '../data/wf/train_NoDef_burst.csv'
    # path_ow = data_folder + 'train_NoDef_UnMon.csv'
    # outpath = data_folder + 'train_NoDef_mix.csv'
    # build_ow_train(path_ow,path_cw,outpath)



    #*************************************
    "balanced processed source csv to burst csv, (function overlapped with part of merge())"
    # data_folder = '../data/wf_ow/'
    # path = data_folder + 'data_NoDef_balanced.csv'
    # out_path = data_folder + 'data_NoDef_burst.csv'
    # slice_threshold = 512
    # tranform2burst(path,out_path,slice_threshold)

    # *************************************
    "Nomalize the burst data to [-1,1] and write it in csv"
    # data_folder = '../data/wf_ow/'
    # path = data_folder + 'data_NoDef_burst.csv'
    # out_path = data_folder + 'data_norm_NoDef_burst.csv'
    # normalize_data(path,out_path)

    # *************************************
    "split data into train/test data"
    # data_folder = '../data/wf_ow/'
    # path = data_folder + 'data_NoDef_burst.csv'
    # out_train_path = data_folder + 'train_NoDef_burst.csv'
    # out_test_path = data_folder + 'test_NoDef_burst.csv'
    # test_train_split(path,out_train_path,out_test_path,test_size=0.2)
