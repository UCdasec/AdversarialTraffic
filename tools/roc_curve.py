"draw roc: fpr-tpr"

import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    data = pd.read_csv(path,encoding='latin-1')
    print(data.head(2))

    tpr = data['TPR']
    fpr = data['FPR']

    return tpr,fpr


def roc(data):
    "data = [fpr_cnn,tpr_cnn,fpr_lstm,tpr_lstm]"
    marker_size = 18
    fontsize = 18

    plt.figure(num=1,figsize=(8, 6), dpi=150)
    plt.plot(data[0],data[1],color='red', marker='*',linestyle='--',linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size,label='CNN')
    plt.plot(data[2],data[3],color='blue', marker='^',linestyle='--',linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size,label='LSTM')
    plt.xlabel('False Positive Rate',{'size':fontsize})
    plt.ylabel('True Positive Rate',{'size':fontsize})
    # plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.tick_params(labelsize=fontsize)
    plt.legend(loc='best',fontsize=fontsize)

    plt.savefig('../fig/roc_E8.eps')
    plt.show()






def roc_main():
    "read csv"
    classifiers = ['cnn','lstm']  #[cnn,lstm]

    data = []
    for classifier in classifiers:
        path = '../result/roc/experiment8/ow_results_' + classifier + '.csv'
        # path = '../result/roc/experiment9/ow_results_' + classifier + '_gan.csv'
        tpr,fpr = load_data(path)
        data += [fpr,tpr]

    "draw roc"
    roc(data)





if __name__ == '__main__':
    roc_main()