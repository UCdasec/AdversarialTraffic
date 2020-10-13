import os
os.sys.path.append('..')
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, VarianceThreshold
import glob,random
import matplotlib.pyplot as plt
import seaborn as sns
from train import utils_wf
import numpy as np




def models(X,y):
    svm_class = SVC()
    rf_class = RandomForestClassifier(n_estimators = 10)
    log_class = LogisticRegression()
    abc_default = AdaBoostClassifier(n_estimators=50,learning_rate=1)   #abs use decision tree as default weak base learner
    abc_rf = AdaBoostClassifier(n_estimators=50,base_estimator=rf_class,learning_rate=1)
    abc_svm = AdaBoostClassifier(n_estimators=50,base_estimator=SVC(probability=True, kernel='linear'),learning_rate=1)
    models = [svm_class,rf_class,log_class,abc_default,abc_rf,abc_svm]
    model_name = ['SVM','RandomFrest','LogisticRegression','abc_default','abc_rf','abc_svm']
    for i in range(len(models)):
        score_model = cross_val_score(models[i],X,y,scoring = 'accuracy', cv=10).mean()
        print('average accuracy of {} model is {}'.format(model_name[i],score_model))
        print('\n')



def selected_classifiers(x_train,y_train,x_test,y_test):
    # x_train,_,y_train,_ = train_test_split(X,y, test_size = 0.2,stratify=y)
    svm_class = SVC(gamma='auto')
    d_tree = DecisionTreeRegressor(max_depth=8)
    knn = neighbors.KNeighborsClassifier(n_neighbors=200,weights='uniform')
    rf_class = RandomForestClassifier(n_estimators=10)
    abc_rf = AdaBoostClassifier(n_estimators=50, base_estimator=rf_class, learning_rate=1)
    abc_default = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    models = [knn,svm_class,rf_class, abc_rf, abc_default]
    models_name =['knn','svm','rf', 'abc_rf', 'abc_default']
    class_names = ['FGSM-AdvTraffic','DeepFool-AdvTraffic','PGD-AdvTraffic','AdvGAN-AdvTraffic']

    for i,model in enumerate(models):
        print('model name %s' % models_name[i])
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print ('\n', '{} model '.format(models_name[i]))
        print (classification_report(y_test,y_pred))
        print ('confusion matrix is {}'.format(confusion_matrix(y_test,y_pred)))
        print ('accuracy is {}'.format(metrics.accuracy_score(y_test,y_pred)))

        # plot confusion matrix figure: one method
        # fontsize = 15
        # # plt.figure(figsize=(8, 6), dpi=150)
        # plt.rcParams.update({'font.size': 16})
        # disp = plot_confusion_matrix(model, x_test, y_test,
        #                              # display_labels=class_names,
        #                              cmap=plt.cm.Blues,
        #                              values_format= '.2f',
        #                              normalize='true')
        # # plt.tick_params(labelsize=fontsize)
        # plt.xlabel('Predicted Label', {'size': fontsize})
        # plt.ylabel('True Label', {'size': fontsize})
        # plt.savefig('../fig/%s_confusion.eps' % models_name[i])
        # print(disp.confusion_matrix)
        # plt.show()

        cm = confusion_matrix(y_test,y_pred)
        heat_map(cm,models_name[i],namorlize=True)



def plot_cm(model,x_test,y_test,figure_name):
    "plot confusion matrix figure"

    fontsize = 15
    # plt.figure(figsize=(8, 6), dpi=150)
    plt.rcParams.update({'font.size': 16})
    disp = plot_confusion_matrix(model, x_test, y_test,
                                 # display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 values_format='.2f',
                                 normalize='true')
    # plt.tick_params(labelsize=fontsize)
    plt.xlabel('Predicted Label', {'size': fontsize})
    plt.ylabel('True Label', {'size': fontsize})
    plt.savefig('../fig/%s_confusion.eps' % figure_name)
    print(disp.confusion_matrix)
    plt.show()



def heat_map(cm,fig_name,namorlize=False):

    if namorlize:
        cm_new = []
        for i in range(len(cm)):
            row = cm[i,:]
            temp = [x/sum(row) for x in row]
            cm_new.append(temp)
        cm = cm_new

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt=".2f",annot_kws={'size':16});
    # Labels, title and ticks
    label_font = {'size': '15'}  # Adjust to fit
    ax.set_xlabel('Predicted Label', fontdict=label_font);
    ax.set_ylabel('True Label', fontdict=label_font);

    #color bar font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    #tick font size
    ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
    # y axis tick without rotation
    plt.yticks(rotation=0)
    plt.savefig('../fig/%s_cm.eps' % fig_name)

    plt.show()



def main_cm_manual(fig_name):
    "customized confusion matrix for random forest"

    # cm results of random forest
    cm_rf = np.array([[0.97,0.03,0.00,0.00],[0.05,0.94,0.01,0.00],[0.01,0.01,0.98,0.00],[0.00,0.00,0.00,1.00]])
    cm_adaboost = np.array([[0.66,0.30,0.04,0.00],[0.32,0.63,0.05,0.00],[0.03,0.02,0.95,0.00],[0.00,0.00,0.00,1.00]])
    # cm_adab_rf = np.array([[3301,455,5,0],[193,3568,6,5],[45,20,3808,0],[3,17,6,3768]])
    cm_adab_rf = np.array([[0.88,0.12,0.00,0.00],[0.05,0.95,0.00,0.00],[0.01,0.01,0.98,0.00],[0,0.01,0,0.99]])
    # cm_cnn = np.array(([[2606,32,30,2082],[555,83,22,4090],[2484,48,54,2164],[1,1,0,4748]]))
    cm_cnn = np.array(([[0.54,0.01,0.01,0.44],[0.12,0.02,0.00,0.86],[0.52,0.01,0.01,0.46],[0.00,0.00,0.00,1.00]]))

    # draw figure
    heat_map(cm_adab_rf,fig_name,namorlize=True)


def main(opts):

    x_train,y_train = utils_wf.load_csv_data(opts['filepath'])

    test_x,test_y = utils_wf.load_csv_data(opts['test_filepath'])

    selected_classifiers(x_train,y_train,test_x,test_y)


def get_opts():
    return{
        'filepath':'../data/wf/cnn/adv_train_all.csv',
        'test_filepath':'../data/wf/cnn/adv_test_all.csv',

    }


if __name__ == '__main__':
    opts = get_opts()

    # for random forest
    # figname = 'adboost_rf_cm'
    # main_cm_manual(figname)


    main(opts)
