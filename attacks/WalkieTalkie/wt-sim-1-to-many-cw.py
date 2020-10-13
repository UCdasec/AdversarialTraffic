'''
This script performs WT simulation with 1-to-1 mappings.

Burst sequence of site A: b_1, b_2, ... b_n
Burst sequence of site B: c_1, c_2, .... c_m

The molding would be burst by burst pairs such as (b_1, c_1), (b_2, c_2), ....., (b_n, c_m)

assume that b_1 has 3 packets {+1 +1 +1} and c_1 has 5 packets  {+1 +1 +1 +1 +1}

finally we do the molding as {+1 +1 +1 +1 +1} , 3 original packets and 2 dummy packets.


To run this script you will have to make following changes to the code:

1. BURST_PATH: directory containing your burst files
2. BURST_DEF_PATH: directory to save your defended burst files
3. CSV_PATH: path to save your csv file for traces and decoy mapping information
4. nb_sites: number of websites you have in your dataset or classes
5. nb_sites_inst: number of instances of each website
'''

import os
import sys
import csv
import random
random.seed(7)
import pandas as pd


# path to burst files (directory)
# BURST_PATH = '/home/danijy/final/webfp/closed-world/data/no-def-keywords/burst/'
BURST_PATH = '../../data/WalkieTalkie/batch/%s/' % 'train'

# path to save defended burst files (directory)
# BURST_DEF_PATH = '/home/danijy/final/webfp/closed-world/data/wt-defense-keyword/wt-def-burst/'
BURST_DEF_PATH = '../../data/WalkieTalkie/defended_batch/%s' % 'train'
if not os.path.isdir(BURST_DEF_PATH):
    print('Creating directory for saving w-t defended burst.')
    os.makedirs(BURST_DEF_PATH)
else:
    print('Directory already available for saving w-t defended traces.')

# path to save csv files (directory)
# CSV_PATH = '/home/danijy/final/webfp/closed-world/data/wt-defense-keyword/csvs/'
CSV_PATH = '../../data/WalkieTalkie/defended_csv/'
# saving traces to csv after defending them with WT mechanism
if not os.path.isdir(CSV_PATH):
    print('Creating directory for saving non-defended burst.')
    os.makedirs(CSV_PATH)
else:
    print('Directory already available for saving non-defended traces.')

nb_sites = 95  # Total number of websites in dataset
nb_sites_inst = 368  # Total number of instances of each site in dataset

# Methods required for reading burst files and generating appropriate format
def read_mbursts(fname):
    """
    Reading the burst sequences from the burst files.
    """
    if fname[-6:] != ".burst":
        print("write_mbursts format incorrect")
        sys.exit(0)

    f = open(fname, "r")
    lines = f.readlines()
    f.close()

    mbursts = []

    for li in lines:
        mbursts.append([])
        li = li.split(",")  # last one is "\n"
        li = [float(i) for i in li]
        for p in li:
            mbursts[-1].append(p)

    return mbursts


def mbursts_to_mpairs(mbursts):
    """
    #mpair just coalesces mburst
    Counts the incoming and outgoing packets in each cell sequence
    """
    mpairs = []

    for burst in mbursts:
        out_count = burst.count(1)
        in_count = burst.count(-1)
        mpairs.append([out_count, in_count])

    return mpairs


def equal_seq_1(real_page, decoy_page):
    """
    This method checks if the length of burst sequences for both real and decoy pages
    """
    dummy = [0, 0]
    #     print('real-page: ', len(real_page))
    #     print('decoy-page: ', len(decoy_page))
    count = 0
    if len(decoy_page) < len(real_page):
        count = abs(len(real_page) - len(decoy_page))

        for i in range(count):
            decoy_page.append(dummy)

    #     print('decoy-page after adding dummy packets: ', len(decoy_page))

    return real_page, decoy_page


def equal_seq_2(real_page, decoy_page):
    """
    This method checks if the length of burst sequences for both real and decoy pages
    """
    dummy = [0, 0]
    #     print('real-page: ', len(real_page))
    #     print('decoy-page: ', len(decoy_page))
    count = 0
    if len(decoy_page) < len(real_page):
        count = abs(len(real_page) - len(decoy_page))

        for i in range(count):
            decoy_page.append(dummy)

    #     print('decoy-page after adding dummy packets: ', len(decoy_page))

    return real_page, decoy_page


def defended_seq(real_page, decoy_page):
    """
    This method generates defended sequence from real and decoy page
    """
    seq = []
    #     print('real page: ', real_page)
    #     print('decoy page: ', decoy_page)
    for k in range(len(real_page)):
        temp_r = real_page[k]
        temp_d = decoy_page[k]
        out_p = [temp_r[0], temp_d[0]]
        in_p = [temp_r[1], temp_d[1]]
        b = [max(out_p), max(in_p)]
        seq.append(b)
    return seq


def mpairs_to_mbursts(seq):
    """
    This method converts pairs of defended to set to burst
    """
    dire = []
    # Creating direction sequence of 1 for outgoing and -1 for incoming
    for i in range(len(seq)):
        temp = seq[i]
        p = []
        for i in range(temp[0]):
            p.append(1.0)
        for j in range(temp[1]):
            p.append(-1.0)
        dire.append(p)
    return dire

# loading sequences
print('loading the sequences ...')
data = []  # list containing all the data
for i in range(0, nb_sites):
    data.append([])
    for j in range(0, nb_sites_inst):
        fname = str(i) + "-" + str(j) + ".burst"
        # print('Processing file: ', fname)
        mbursts = read_mbursts(BURST_PATH + fname)
        # print(mbursts)
        data[-1].append(mbursts_to_mpairs(mbursts))
        # print(data[i])
print('sequences loaded successfully.')

all_sites = []
for i in range(nb_sites):
    all_sites.append(i)
print('number of sites: ', len(all_sites))
print('all sites are: ', all_sites)

# number of instances for each site
nb_inst = []
for i in range(nb_sites_inst):
    nb_inst.append(i)
print('number of instances for each site: ', len(nb_inst))
print('instance number are: ', nb_inst)


# selecting first five sites as decoy
decoy_sites = all_sites[:50]
print('sites selected for decoy: ', decoy_sites)

nondecoy_sites = all_sites[50:]
print('selected selected for non-decoy: ', nondecoy_sites)

# creating 1-to-many mapping
mapping = []
for i in range(nb_sites):
    decoy_p = random.choice(decoy_sites)
    for j in range(len(nb_inst)):
        #         print('real-page: %d - %d'%(i, j))
        decoy_p_inst = random.choice(nb_inst)
        #         print('decoy-page: %d - %d'%(decoy_p, decoy_p_inst))

        # saving the mapping to list
        mapping.append([i, j, decoy_p, decoy_p_inst])  # real_page, real_page_inst, decoy_page, decoy_page_inst

decoy_sites = all_sites[:50]
print('Saving ground truth data.')
# Saving ground truth
ground_truth = pd.DataFrame(mapping)
ground_truth.columns = ['real_page', 'real_page_inst', 'decoy_page', 'decoy_page_inst']
ground_truth.head()

# Saving ground truth to csv
ground_truth.to_csv(CSV_PATH + 'wt-def-keyword-decoy-mapping.csv', index=False)
print('Ground truth data saved successfully.')

print('Defending sensitive pages with non-sensitive pages.')
for i in range(len(ground_truth)):
    t_burst = []
    real_pg = ground_truth['real_page'][i]
    real_pg_inst = ground_truth['real_page_inst'][i]
    #         print('Real Page i: ', real_pg, '-', real_pg_inst)
    decoy_pg = ground_truth['decoy_page'][i]
    decoy_pg_inst = ground_truth['decoy_page_inst'][i]
    #         print('Decoy page i: ', decoy_pg, '-', decoy_pg_inst)
    # defending the data
    real_page = data[real_pg][real_pg_inst]
    decoy_page = data[decoy_pg][decoy_pg_inst]
    real_page_2, decoy_page_2 = equal_seq_1(real_page, decoy_page)
    sseq = defended_seq(real_page_2, decoy_page_2)
    #         print('lenght-sseq: ', len(sseq))
    #         print('sseq: ', sseq)
    t_burst = mpairs_to_mbursts(sseq)
    # saving the burst to file
    temp_fname = str(real_pg) + '-' + str(real_pg_inst) + '.burst'
    #         print(temp_fname)
    temp_burst = t_burst
    f = open(BURST_DEF_PATH + '/' + temp_fname, 'w')
    for ele in temp_burst:
        print(str(ele).lstrip('[').rstrip(']'), file=f)
    f.close()

print('Defending non-sensitive pages with sensitive pages.')

# reading all the defended files from the directory
wt_def_fs = os.listdir(BURST_DEF_PATH)
print('Total number of files: ', len(wt_def_fs))

# saving the data to csv file
# trace = []
# label = []
# for i in range(len(wt_def_fs)):
#     temp_label = wt_def_fs[i].split('-')[0]
#     label.append(temp_label)
#     temp_f = os.path.join(BURST_DEF_PATH, wt_def_fs[i])
#     all_values = []
#     with open(temp_f, newline='') as infile:
#         reader = csv.reader(infile)
#         for row in reader:
#             all_values.extend(row)
#         infile.close()
#     trace.append(all_values)
#
# wt_def_df = pd.DataFrame(data=trace)
# wt_def_df = wt_def_df.fillna(0.0)
# label_df = pd.DataFrame(data=label, columns=['label'])
# frames = [label_df, wt_def_df]
# all_def_tr = pd.concat(frames, axis=1, join='inner')
#
# # saving data to csv file
# all_def_tr.to_csv(CSV_PATH + 'wt-def-keyword-data.csv', index=False)
# print('defended data saved successfully.')