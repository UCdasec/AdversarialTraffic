from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pandas as pd
import torch
import torch.utils.data as Data
import numpy as np

"""
devide the data into train and test data 
and write it into csv files
"""


def incoming_data_extract(data, slice_threshold=256):
    "remove outgoing data for each traffic"
    "padding to the same length with 0"
    "output dataframe"

    #outgoing package size positice value
    data_group = []
    # temp1 = []
    for i in range(len(data)):
        row_i = data.iloc[i]
        temp = []
        for x in row_i:
            if x < 0:
                temp.append(x)
        # temp1.append(len(temp))
        data_group.append(temp[:slice_threshold])
    df = pd.DataFrame(data_group) # after convert list to DF, all short lists will append with "NaN' to the same length
    df = df.fillna(0)       # all those 'NaN' are padded to same length, replace 'NaN' with 0
    return df




def load_data(path):
    "col_length: column length of each input"

    raw_data = pd.read_csv(path)

    "encode labels"
    y_raw = raw_data.iloc[:,0]
    encoder = preprocessing.LabelEncoder()
    labels = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    labels = np_utils.to_categorical(labels,num_classes=num_classes)

    "processing data"
    X = raw_data.iloc[:,1:]
    X = incoming_data_extract(X)

    "train test data"
    X_train,X_test,y_train,y_test = train_test_split(X,labels,train_size=0.94,shuffle=True,stratify=labels) # stratify = y, split data according to proportion


    return X_train,y_train,X_test,y_test,num_classes


if __name__ == '__main__':
    path = 'generic_class.csv'
    X_train, y_train, X_test, y_test, num_classes = load_data(path)
    # vector to int
    y_train = np.argmax(y_train,1)
    y_test = np.argmax(y_test,1)   #axis=1, row

    X_test['label'] = y_test
    X_train['label'] = y_train
    print(X_train.head())
    X_test.to_csv('traffic_test.csv',index=0)
    X_train.to_csv('traffic_train.csv',index=0)
