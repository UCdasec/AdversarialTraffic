""""
obtain bandwidth: the total noise after defenses,
and visualize it
"""

import os
os.sys.path.append('..')

from train import utils_wf
import numpy as np
import matplotlib.pyplot as plt


class get_bandwith:
    def __init__(self,opts):
        self.opts = opts
        self.data_path = '../data/wf/%s/' % opts['classifier']



    def get_noise(self,x,x_adv):

        "orginal x"
        ave_x = np.mean(np.sum(abs(x),axis=1).tolist())

        "noise"
        noise = x_adv - x
        tt = abs(noise)
        bandwidth = np.sum(tt,axis=1).tolist()

        # "other way to obtain average bandwidth, results differ"
        # bandwidth_all = np.sum(tt,axis=1)
        # len_all = np.sum(abs(x),axis=1)
        # average_bw = np.mean(bandwidth/len_all)
        # print("average bandwidth overhead: ", average_bw)

        max1 = max(bandwidth)
        min1 = min(bandwidth)
        # print(bandwidth.count(0))
        ave_bandwidth = np.mean(bandwidth)
        print('max bandwidth', max1)
        print('min bandwidth',min1)
        print('average perturbation size:',ave_bandwidth)
        print('average burst length {}'.format(ave_x))
        print('ave_pert_size/ave_trace_size = ',ave_bandwidth/ave_x)


        return ave_bandwidth,bandwidth


    def get_statistic(self,data):
        "get statistic info of different certian range, return data for pie figure"

        r1,r2,r3,r4,r5,r6 = [],[],[],[],[],[]

        for i,x in enumerate(data):
            if x >= 0 and x < 250:
                r1.append(x)
            elif x >= 250 and x < 500:
                r2.append(x)
            elif x >= 500 and x < 750:
                r3.append(x)
            elif x >= 750 and x < 1000:
                r4.append(x)
            elif x >= 1000 and x < 1250:
                r5.append(x)
            elif x >= 1250 :
                r6.append(x)

        stat = [r1,r2,r3,r4,r5,r6]
        len_data = [len(r1),len(r2),len(r3),len(r4),len(r5),len(r6)]
        print(len_data)
        return len_data


    def visualize(self,data,i):
        "pie figure"
        plt.figure(figsize=(9,9))
        labels = ['[0,250)','[250,500)','[500,750)','[750,1000)','[1000,1250)','>1250']
        sizes = data
        plt.pie(sizes,labels=labels,autopct = '%3.2f%%',shadow=True)
        plt.legend()
        plt.title(self.opts['adversary_list'][i] + ' bandwidth distribution')
        plt.savefig('../fig/bandwidth_ditribution_' + self.opts['adversary_list'][i] +'.eps')
        plt.show()


    def result(self):
        x, y = utils_wf.load_csv_data(self.opts['filepath'])

        for i,f in enumerate(self.opts['adv_file_list']):
            print('-' * 30)
            print(f)
            if f == 'adv_test_WT.csv':
                path = '../data/WalkieTalkie/defended_csv/'
                x_adv, y_adv = utils_wf.load_csv_data(path + f)
            else:
                x_adv, y_adv = utils_wf.load_csv_data(self.data_path + f)
            ave_bandwidth,bandwidth = self.get_noise(x,x_adv)
            print('{} bandwith average overhead is {}'.format(self.opts['adversary_list'][i],ave_bandwidth))

            len_data = self.get_statistic(bandwidth)

            self.visualize(len_data,i)



def main(opts):
    bandwith_overhead = get_bandwith(opts)
    bandwith_overhead.result()



def get_opts(classifier):
    return{
        'classifier':classifier,
        #---------"test data"
        'filepath': '../data/wf/test_NoDef_burst.csv',
        'adv_file_list': ['adv_test_FGSM.csv','adv_test_DeepFool.csv','adv_test_PGD.csv','adv_test_GAN.csv','adv_test_WT.csv'],
        
        #---------"train data"
        # 'filepath': '../data/wf/train_NoDef_burst.csv',
        # 'adv_file_list': ['adv_train_FGSM.csv','adv_train_DeepFool.csv','adv_train_PGD.csv','adv_train_GAN.csv'],

        'adversary_list':['FGSM','DeepFool','PGD','GAN','WT'],
    }


if __name__ == '__main__':
    classifier = 'lstm'
    opts = get_opts(classifier)
    main(opts)