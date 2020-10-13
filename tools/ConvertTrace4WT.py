"""
# 1
convert burst trace back to the packet trace in binary format, which only consider direction
e.g., [2,-3,2,-1] --> [1,1,-1,-1,-1,1,1,-1]
# 2
transfer the packet trace into certain format to apply WalkieTalkie defense, and write in .burst file
e.g., [1,1,-1,-1,-1,1,1,1,-1]
--> 1,1,-1,-1,-1
    1,1,1,-1
# 3
convert defended data with WT into busrst and write it in csn
"""


import pandas as pd
from train import utils_wf
import os,csv



def load_csv_data(path):
    "col_length: column length of each input"

    raw_data = pd.read_csv(path)
    X = raw_data.iloc[:,:-1]
    label = raw_data['label']

    return X, label



def burst2packet(X):
    """
    convert burst trace to trace with packet direction
    return packet direction trace in list
    e.g., [2,-3,2,-1] --> [1,1,-1,-1,-1,1,1,-1]
    """

    "dataframe to list"
    if type(X) == list:
        pass
    else:
        X = X.values.tolist()

    trace = []
    for x in X:
        trace_p = []
        for p in x:
            if p < 0:
                sign = -1
            else:
                sign = 1
            trace_p += [1.0*sign for e in range(int(abs(p)))]
        trace.append(trace_p)

    return trace


def packet2cetainBurst(X,y,data_type):
    """
    trace with packet direction transform to burst format
    that can be applied with WalkieTalkie defense
    e.g., [1,1,-1,-1,-1,1,1,1,-1]
    --> 1,1,-1,-1,-1
    1,1,1,-1
    :param X: trace with packet direction
    :param y: label
    :return:
    """

    y_dic = {}

    for i,x in enumerate(X):

        # get the id of the trace give each class
        if y[i] not in y_dic:
            y_dic[y[i]] = 0
        else:
            y_dic[y[i]] += 1

        # create a bust file for writing
        out_path = '../data/WalkieTalkie/batch/%s/%d-%d.burst' % (data_type,y[i],y_dic[y[i]])
        f = open(out_path,'w+')

        # devide the trace in certain format
        count_sign = 0
        flag_0 = True
        for j in range(len(x)-1):
           if x[j] * x[j+1] < 0:
               count_sign += 1
           if count_sign == 2 and flag_0:
               flag_0 = False
               for n,e in enumerate(x[:j+1]):
                   if n == j:
                       f.write(str(e))
                   else:
                       f.write(str(e) +',')
               f.write('\n')
               position = j + 1
               count_sign = 0
           elif count_sign == 2 and not flag_0:
               for m,k in enumerate(x[position:j+1]):
                   if m == j-position:
                       f.write(str(k))
                   else:
                       f.write(str(k) + ',')
               f.write('\n')
               position = j+1
               count_sign = 0
        f.close()
        print('file saved on %s' % out_path)
    print(y_dic)



def load_burst_file(path):
    """
    load data from burst file and return each trace as list
    :param path:
    :return:
    """
    print(path)
    data = []
    with open(path,'r') as f:
        lines = f.readlines()
    for line in lines:
        for x in line.split(','):
            # print(x,type(x))
            try:
                x = float(x)
            except ValueError:
                pass
            data.append(x)
    return data


# def load_burst_file(path):
#     """
#     load data from burst file and return each trace as list
#     :param path:
#     :return:
#     """
#     data = []
#     with open(path,newline='') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             data.extend(row)
#     return data






def main2burst():
    "convert csv file to burst file"
    data_type = ['test','train']
    for type in data_type:
        path = '../data/wf/%s_NoDef_burst.csv' % type
        print('working directory',path)
        X,y = load_csv_data(path)
        x_data = burst2packet(X)
        packet2cetainBurst(x_data,y,type)



def main2csv():
    "convert burst file to csv file"
    data_type = ['test','train']
    for type in data_type:
        path = '../data/WalkieTalkie/defended_batch/%s/' % type
        out_path = '../data/WalkieTalkie/defended_csv/adv_%s_WT.csv' % type
        items = os.listdir(path)
        data_list = []
        labels = []
        for item in items:
            if item[-6:] == '.burst':
                label = int(item.split('-')[0])
                data = load_burst_file(path+item)
                data_list.append(data)
                labels.append(label)

        # tt = pd.DataFrame(data_list)
        # tt['label'] = labels
        # tt.to_csv('../data/WalkieTalkie/defended_csv/orig_%s.csv' % type,index=0)

        "binary to burst"
        data_burst,data_burst_noSlice = utils_wf.burst_transform(data_list,slice_threshold=512)
        data_df = utils_wf.convert2dataframe(data_burst_noSlice,labels,mode='padding')
        utils_wf.write2csv(data_df,out_path)
        print('{} ... saved successfully.'.format(out_path))




if __name__ == '__main__':

    # convert cvs to burst
    # main2burst()

    # convert burst to csv
    main2csv()

