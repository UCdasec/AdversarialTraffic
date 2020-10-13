'''
This script converts traces in CSV file to burst format, which is further used for defending them using
Walkie-Talkie defense mechanism.
'''

import os
import pandas as pd
from datetime import datetime

def split_traffic_trace(raw_trace):
    """
    This function gets the indices from which the traffic trace is to be splitted.
    """
    ind = []
    for i in range(len(raw_trace)):
        if raw_trace[i] == -1.0:
            if (raw_trace[i] == -1.0 and raw_trace[i+1] == 1.0):
                ind.append(i)
    return ind

def split_traffic_trace_1(raw_trace):
    """
    This function gets the indices from which the traffic trace is to be splitted.
    """
    ind = []
    for i in range(len(raw_trace)):
        if raw_trace[i] == 1.0:
            if (raw_trace[i] == 1.0 and raw_trace[i+1] == -1.0):
                ind.append(i)
    return ind

def burst_seq(seq, indices):
    startpos = 0
    for index in indices:
        yield seq[startpos:index+1]
        startpos = index + 1
    yield seq[startpos:]

def burst_seq_final(t):
    """
    This forms the burst sequence from the traffic trace.
    """
    final_burst = []
    for i in range(len(t)):
        tmp = t[i]
        final_burst.append(tmp)
    return final_burst

def remove_zeros(lst):
    """
    This method removed zeros for the last burst.
    """
    temp_burst = lst
    last_bst = temp_burst.pop(-1)
    tmp = list(filter(lambda num: num != 0.0, last_bst))
    temp_burst.append(tmp)
    return temp_burst

# Path for data csv file
DATA_PATH = '/home/danijy/final/webfp/closed-world/data/no-def-keywords/csvs/keyword-100-classes-5000-direction.csv'
# Path for saving burst files
BURST_PATH = '/home/danijy/final/webfp/closed-world/data/no-def-keywords/burst/'
if not os.path.isdir(BURST_PATH):
    print('Creating directory for saving non-defended burst.')
    os.mkdir(BURST_PATH)
else:
    print('Directory already available for saving non-defended traces.')

# Reading the csv files from the directory
print('loading the data from csv files ...')
all_data = pd.read_csv(DATA_PATH)
print('csv files loaded successfully.')

# Getting unique number of labels from the dataset
nb_labels = all_data['label'].nunique()
print('number of unique labels: ', nb_labels)

# Removing all the labels with count less than 90
all_dir_data = all_data.groupby('label').filter(lambda x: len(x)>=250)

# Selecting equal number of samples from each class
n = 250 # number of rows for particular class
grouped = all_data.groupby(['label'], as_index=False)
all_dir_data = grouped.apply(lambda frame: frame.sample(n))
all_dir_data.index = all_dir_data.index.droplevel(0)

# grouping the dataframe by class labels
data_grp = all_dir_data.groupby(['label'])

file_names = []
burst_data = []
for i in range(len(data_grp.groups.keys())):
    print('processing label:', i)
    lbl = data_grp.get_group(i).drop(['label'], axis = 1)
    # temp_lbl = str(data_grp.get_group(i).pop('label'))
    for j in range(len(lbl)):
        fname = str(i) + '-' + str(j) + '.burst'
        temp_trace = lbl.iloc[j].values.tolist()
        # print(temp_trace)

        if temp_trace[-1] != 0.0:
            temp_trace.append(0.0)

        if temp_trace[0] == 1.0:
            s_ind = split_traffic_trace_1(temp_trace)

        if temp_trace[0] == -1.0:
            s_ind = split_traffic_trace(temp_trace)

        s_ind = split_traffic_trace(temp_trace)
        t = list(burst_seq(temp_trace, s_ind))
        final_burst = burst_seq_final(t)
        final_burst = remove_zeros(final_burst)
        burst_data.append(final_burst)
        file_names.append(fname)

for i in range(len(burst_data)):
    temp_fname = file_names[i]
    temp_burst = burst_data[i]
    f = open(BURST_PATH + '/' + temp_fname, 'w')
    for ele in temp_burst:
        print(str(ele).lstrip('[').rstrip(']'), file = f)
    f.close()
print('burst files successfully at', BURST_PATH)
print('program execution completed.')