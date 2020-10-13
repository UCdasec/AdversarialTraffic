"draw recall-precision"

import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    """
    load data
    """
    data = pd.read_csv(path)
    print(data.head(2))

    return data


def recall_precision(data,outpath):
    """
    [Precision-fgsm,Recall-fgsm,Precision-deepfool,Recall-deepfool,
    Precision-pgd,Recall-pgd, Precision-GAN,Recall-GAN]
    """

    header = ['P_fgsm','R_fgsm','P_deepfool','R_deepfool',
    'P_pgd','R_pgd','P_GAN','R_GAN']

    marker_size = 18
    fontsize = 18

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(data[header[0]], data[header[1]], color='red', marker='+', linestyle='--', linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size,label='FGSM-AdvTraffic')
    plt.plot(data[header[2]], data[header[3]], color='blue', marker='^', linestyle='--', linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size,label='DeepFool-AdvTraffic')
    plt.plot(data[header[4]], data[header[5]], color='green', marker='o', linestyle='--', linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size,label='PGD-AdvTraffic')
    plt.plot(data[header[6]], data[header[7]], color='black', marker='*', linestyle='--', linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size,label='AdvGAN-AdvTraffic')
    plt.xlabel('Recall', {'size': fontsize})
    plt.ylabel('Precision', {'size': fontsize})
    plt.tick_params(labelsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)

    plt.savefig(outpath)
    plt.show()



def main():

    classifiers = ['cnn','lstm']  #[cnn,lstm]


    for classifier in classifiers:
        path = '../result/roc/experiment9/ow_results_' + classifier + '.csv'
        data = load_data(path)
        outpath = '../fig/rec_pre_%s.eps' % classifier
        recall_precision(data,outpath)




if __name__ == '__main__':
    main()