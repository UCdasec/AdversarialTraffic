import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler



def params_cnn(num_class,input_size):
    "hyperparameters for target model"

    print('input_size: {}'.format(input_size))

    return {
        'conv1_input_channel': 1,
        'conv2_input_channel': 128,
        'conv3_input_channel': 128,
        'conv4_input_channel': 64,
        'conv1_output_channel': 128,
        'conv2_output_channel': 128,
        'conv3_output_channel': 64,
        'conv4_output_channel': 256,
        'kernel_size1': 7,
        'kernel_size2': 19,
        'kernel_size3': 13,
        'kernel_size4': 23,
        'stride1': 1,
        'stride2': 1,
        'stride3': 1,
        'stride4': 1,
        'padding1': 3,
        'padding2': 9,
        'padding3': 6,
        'padding4': 11,
        'drop_rate1': 0.1,
        'drop_rate2': 0.3,
        'drop_rate3': 0.1,
        'drop_rate4': 0.0,
        'pool1': 2,
        'pool2': 2,
        'pool3': 2,
        'pool4': 2,
        'num_classes': num_class,
        'input_size':input_size
    }


def params_cnn_detect(num_class,input_size):
    "hyperparameters for target model of detection"

    print('input_size: {}'.format(input_size))

    return {
        'conv1_input_channel': 1,
        'conv2_input_channel': 64,
        'conv1_output_channel': 64,
        'conv2_output_channel': 128,
        'kernel_size1': 3,
        'kernel_size2': 3,
        'stride1': 1,
        'stride2': 1,
        'padding1': 1,
        'padding2': 1,
        'drop_rate1': 0.1,
        'drop_rate2': 0.3,
        'pool1': 2,
        'pool2': 2,
        'num_classes': num_class,
        'input_size':input_size
    }


def params_lstm(num_class,input_size,batch_size):
    "hyperparameters for target model"

    print('input_size: {}'.format(input_size))
    # orignal result based on the 3 layers and 1024 hidden dim, 0.1 dropout

    # searched one: 2 layers and 256 hidden dim, 0.4 dropout, tanh, lr =0.006
    return {
        'num_layers':3,
        'hidden_dim':1024,
        'dropout': 0.1,
        'activation_fun':'selu',
        'optimizer':'Adam',
        'batch_size': batch_size,
        'batch_first':True,
        'bidirectional':False,
        'num_classes': num_class,
        'input_size':input_size
    }


def params_lstm_kf(num_class,input_size,batch_size):
    "hyperparameters for target model in k-fold validation, NNI searched"

    print('input_size: {}'.format(input_size))

    return {
        'num_layers':2,
        'hidden_dim':256,
        'dropout': 0.4,
        'activation_fun':'tanh',
        'optimizer':'Adam',
        'batch_size': batch_size,
        'batch_first':True,
        'bidirectional':False,
        'num_classes': num_class,
        'input_size':input_size
    }



def params_lstm_ow(num_class,input_size,batch_size):
    "hyperparameters for target model in open world setting"

    print('input_size: {}'.format(input_size))

    return {
        'num_layers':2,
        'hidden_dim':256,
        'dropout': 0.1,
        'activation_fun':'elu',
        'optimizer':'Adam',
        'batch_size': batch_size,
        'batch_first':True,
        'bidirectional':False,
        'num_classes': num_class,
        'input_size':input_size
    }


def params_lstm_ow_eval(num_class,input_size,batch_size):
    "hyperparameters for target model in open world setting"

    print('input_size: {}'.format(input_size))

    return {
        'num_layers':2,
        'hidden_dim':256,
        'dropout': 0,
        'activation_fun':'elu',
        'optimizer':'Adam',
        'batch_size': batch_size,
        'batch_first':True,
        'bidirectional':False,
        'num_classes': num_class,
        'input_size':input_size
    }



def params_lstm_eval(num_class,input_size,batch_size):
    "eval: lstm can't work on eval model, so set dropout as 0 to make train model as eval model"

    print('input_size: {}'.format(input_size))

    return {
        'num_layers':3,
        'hidden_dim':1024,
        'dropout': 0,
        'activation_fun': 'tanh',
        'batch_size': batch_size,
        'batch_first':True,
        'bidirectional':False,
        'num_classes': num_class,
        'input_size':input_size
    }


def params_lstm_detect(num_class,input_size,batch_size):
    "hyperparameters for target model for detecting adversary"

    print('input_size: {}'.format(input_size))

    return {
        'num_layers':1,
        'hidden_dim':64,
        'dropout': 0.1,
        'activation_fun':'selu',
        'optimizer':'Adam',
        'batch_size': batch_size,
        'batch_first':True,
        'bidirectional':False,
        'num_classes': num_class,
        'input_size':input_size
    }



def params_rnn(num_class,input_size,batch_size):
    "hyperparameters for target model"

    print('input_size: {}'.format(input_size))

    return {
        'num_layers':1,
        'hidden_dim':256,
        'dropout': 0.4,
        'activation_fun':'elu',
        'optimizer':'Adam',
        'batch_size': batch_size,
        'batch_first':True,
        'bidirectional':False,
        'num_classes': num_class,
        'input_size':input_size
    }


def params_fcnn(num_class,input_size):
    "hyperparameters for target model"

    print('input_size: {}'.format(input_size))

    return {
        'num_classes': num_class,
        'input_size':input_size
    }



def load_pkl_data(path):
    "load pickle file"
    data = pd.read_pickle(path)

    return data



def burst_transform(data,slice_threshold):
    "transform binary format to burst format"

    "dataframe to list"
    if type(data) == list:
        pass
    else:
        data = data.values.tolist()

    burst_data = []
    burst_noSlicing = []

    for x in data:
        burst_i = []
        count = 1
        temp = x[0]

        for i in range(1,len(x)):
            if temp == 0:
                print('traffic start with 0 error')
                break
            elif x[i] == 0:
                burst_i.append(count*temp)
                break
            elif temp == x[i]:
              count += 1
            else:
                burst_i.append(count*temp)
                temp = x[i]
                count = 1
        burst_data.append(burst_i[:slice_threshold])
        burst_noSlicing.append(burst_i)
    return burst_data,burst_noSlicing



def size_distribution(data,output):
    "plot size distribution in different ways"
    "get lenght of each data"
    length = []
    for x in data:
        length.append(len(x))

    "plot cdf"
    hist,bin_edges = np.histogram(length)
    cdf = np.cumsum(hist/sum(hist))
    plt.figure(0)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel('number of bursts')
    plt.ylabel('Fraction')
    plt.savefig('../fig/' + output +'_size_cdf.pdf')
    plt.show()

    "plot cdf and pdf in one fig"
    width = (bin_edges[1] - bin_edges[0]) * 0.8
    plt.figure(1)
    plt.bar(bin_edges[1:], hist / max(hist), width=width, color='#5B9BD5')
    plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
    plt.xlabel('number of bursts')
    plt.ylabel('Fraction')

    plt.savefig('../fig/' + output +'_size_cdf_pdf.pdf')
    plt.show()


    "get statistic info about the length in dictionary format"
    size_dic = {}
    for x in length:
        if x not in size_dic:
            size_dic[x] = 1
        else:
            size_dic[x] +=1

    "obtain fraction of each value in dic"
    keys = size_dic.keys()
    values = np.array(list(size_dic.values()))
    values_fraction = np.array(list(values))/sum(values)

    "present in bar figure"
    plt.figure(2)
    plt.bar(keys,values)
    plt.xlabel('number of bursts')
    plt.ylabel('numer of traces')
    plt.savefig('../fig/' + output +'_size_distribution.pdf')
    plt.show()

    return size_dic



def data_preprocess(X,y):
    """
    removing any instances with less than 50 packets
    and the instances start with incoming packet
    """
    X_new = []
    y_new = []
    length = []
    for x in X:
        x = [i for i in x if i != 0]
        length.append(len(x))
    for i,x in enumerate(length):
        if x >= 50 and X[i][0] == 1:
            X_new.append(X[i])
            y_new.append(y[i])

    print('No of instances after processing: {}'.format(len(y_new)))

    return X_new,y_new



def convert2dataframe(x,y,mode='padding',slice_threshold=512):
    "convert list to dataframe format, merge x and y in one df"
    df = pd.DataFrame(x)
    row_length = df.shape

    if mode == 'padding':
        df = df.fillna(0)
    else:
        pass

    # check the size of each row
    if row_length[1] != slice_threshold:
        add_shape = [row_length[0],slice_threshold-row_length[1]]
        temp = np.zeros((add_shape))
        frames = [df,pd.DataFrame(temp)]
        df = pd.concat(frames,axis=1,join='inner',ignore_index=True)

    # print('shape of the data ', df.shape)
    df['label'] = y

    return df



def write2csv(data,outpath):
    " dataframe write to csv"

    data.to_csv(outpath,index=0)



def load_csv_data(path):
    "col_length: column length of each input"

    raw_data = pd.read_csv(path)
    x = raw_data.iloc[:,:-1]
    label = raw_data['label']

    return x, label



# def data_normalize(data):
#     """
#     normalize data with MinMaxScaler [-1,1]
#     :param data: ndarray
#     :return: ndarray
#     """
#
#     if isinstance(data,torch.Tensor):
#         data = data.data.cpu().numpy()
#     if not isinstance(data,np.ndarray):
#         data = data.to_numpy()
#
#     input_shape = data.shape
#     data = data.reshape(-1,1)
#     scaler = MinMaxScaler(feature_range=(-1,1))
#     scaler = scaler.fit(data)
#     data = scaler.transform(data)
#     data = data.reshape(input_shape)
#
#     return data


# def inverse_normalize(data):
#     """
#
#     :param data: ndarray
#     :return: ndarray
#     """
#
#     if type(data) == torch.Tensor:
#         data = data.data.cpu().numpy()
#
#     input_shape = data.shape
#     data = data.reshape(-1, 1)
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler = scaler.fit(data)
#     data = scaler.inverse_transform(data)
#     data = data.reshape(input_shape)
#
#     return data


class data_normalize_inverse:
    """
    normalize data with MinMaxScaler [-1,1]
    and inverse the normalized data back
    :param data: ndarray
    :return: ndarray
    """
    def __init__(self,data):

        if isinstance(data, torch.Tensor):
            data = data.data.cpu().numpy()
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        self.input_shape = data.shape
        self.input_data = data.reshape(-1,1)
        self.scaler_handle = MinMaxScaler(feature_range=(-1,1))
        self.scaler = self.scaler_handle.fit(self.input_data)

    def data_normalize(self):
        data_norm = self.scaler.transform(self.input_data)
        data_norm = data_norm.reshape(self.input_shape)

        return data_norm

    def inverse_normalize(self,data_norm):
        input_data = data_norm.reshape(-1,1)
        data = self.scaler.inverse_transform(input_data)
        data = data.reshape(self.input_shape)

        return data



class normalizer:
    """
    normalize data with L1/L2 distance
    and inverse it back
    input data: ndarray,
    """

    def __init__(self, data, mode='l2'):

        if isinstance(data, torch.Tensor):
            data = data.data.cpu().numpy()
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        self.input_shape = data.shape
        self.data = np.transpose(data.squeeze())

        if mode == 'l1':
            self.w = sum(abs(self.data))
        elif mode == 'l2':
            self.w = np.sqrt(sum(self.data ** 2))


    def Normalizer(self):

        x_norm = self.data / self.w
        x_norm = np.transpose(x_norm).reshape(self.input_shape)

        return x_norm,self.w


    def inverse_Normalizer(self, x_norm):

        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.data.cpu().numpy()
        if not isinstance(x_norm, np.ndarray):
            x_norm = x_norm.to_numpy()

        x_norm = np.transpose(x_norm.squeeze())
        x = x_norm * self.w
        x = np.transpose(x).reshape(self.input_shape)

        return x



def round_data(data):
    """
    add round function to keep data in integer
    :param data: ndarray
    :return: ndarray
    """
    data = np.around(data)

    return data


def Data_loader(x, y,batch_size,shuffle=False):
    """
    input: X, y ( X: dataframe, y: ndarray)
    output: tensor_loader. train_loader, test_loader
    """


    " add dimension, df to ndarray "
    x = np.expand_dims(x, axis=1)

    "ndarray to tensor:"
    x_tensor = torch.Tensor(x)
    y_tensor = torch.Tensor(y).long()

    "build input dataformat in pytorch"
    dataset = Data.TensorDataset(x_tensor,y_tensor)

    data_loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
    )

    return data_loader



def load_data_main(path,batch_size,shuffle=False):
    x,y = load_csv_data(path)

    "normalize data"
    # normalization = normalizer(x,'l2')
    # x,_ = normalization.Normalizer()


    data_loader = Data_loader(x,y,batch_size,shuffle)

    return data_loader



def load_NormData_main(path,batch_size):
    "load data after it normalized"

    x,y = load_csv_data(path)

    "normalize data with l1/l2 distance"
    normalization = normalizer(x,'l2')
    x,_ = normalization.Normalizer()


    data_loader = Data_loader(x,y,batch_size)

    return data_loader



def extract_data_each_class(path,num):
    "extract num instances for each class,return data in list"

    X,Y = load_csv_data(path)
    X_new, Y_new = [],[]

    Y_set = set(Y)

    for y_set_i in Y_set:
        temp = y_set_i
        count = 1
        for i,y in enumerate(Y):
            if y == temp and count <= num:
                X_new.append(X.iloc[i,:])
                Y_new.append(y)
                count += 1
                if count > num:
                    break
    print('{} instances per class, {} classes in toal.'.format(num,len(Y_set)))
    return X_new,Y_new


def extract_data_each_class_ow(path,num):
    "extract num instances for each class,return data in list"
    "for open world data, include them all"

    X,Y = load_csv_data(path)
    X_new, Y_new = [],[]

    Y_set = set(Y)

    for y_set_i in Y_set:
        "include all open world data"
        if y_set_i == 95:
            temp = y_set_i
            items = 0
            for i, y in enumerate(Y):
                if y == temp:
                    X_new.append(X.iloc[i, :])
                    Y_new.append(y)
                    items += 1
        else:
            "build balanced data for closed world data"
            temp = y_set_i
            count = 1
            for i,y in enumerate(Y):
                if y == temp and count <= num:
                    X_new.append(X.iloc[i,:])
                    Y_new.append(y)
                    count += 1
                    if count > num:
                        break
    print('{} instances per class of closed world data, {} classes in toal.'.format(num,len(Y_set)-1))
    print('{} instances per class of opened world data, {} classes in toal.'.format(items,1))
    return X_new,Y_new


def get_ow_data(path):
    "get opened world data"

    X, Y = load_csv_data(path)
    X_new, Y_new = [], []

    temp_label = 95
    count = 0
    for i, y in enumerate(Y):
        if y == temp_label:
            X_new.append(X.iloc[i, :])
            Y_new.append(y)
            count += 1
    return X_new,Y_new



def add_perturbation(x,pert):
    """
    the perturbation add to burst format data in Website Finerprinting
    must be keep the result increase in the original direction
    e.g., x = [-2,3,-1], pert = [-0.03,-0.04,-0.1], x+pert=[-2-0.03,3+0,-1-0.1]=[-2.03,3,-1.1]

    :param x: ndarray
    :param pert: ndarray
    :return: torch.Tensor
    """

    input_shape = x.shape

    "Tensor to ndarray"
    if type(x) == torch.Tensor:
        x = x.data.cpu().numpy()
    if type(pert) == torch.Tensor:
        pert = pert.data.cpu().numpy()


    result = []

    for i in range(len(x)):
        for j in range(len(x[i][0])):
            a = x[i][0][j]
            b = pert[i][0][j]
            if a * b >= 0:
                temp = a + b
                result.append(temp)
            else:
                temp = a
                result.append(temp)

    result = torch.Tensor(np.array(result))
    result = result.view(input_shape)

    return result


# def get_advX_wf_main(x,pert, pert_box):
#     """
#     wrap all steps to get adversarial x given website fingerprinting
#     x: ndarray
#     pert: torch.Tensor
#     return: Tensor
#     """
#
#     if isinstance(type(x),torch.Tensor):
#         x = x.data.cpu().numpy()
#
#     "load data normalization handle"
#     norm_inverse = data_normalize_inverse(x)
#
#     "normalize data"
#     x_norm = norm_inverse.data_normalize()
#
#     "clamp perturbation"
#     pert = torch.clamp(pert, -pert_box, pert_box)
#
#     "add pert to x given the restrictions"
#     adv_x = add_perturbation(x_norm, pert)
#
#     "inverse the normalized data"
#     adv_x = norm_inverse.inverse_normalize(adv_x)
#
#     "add round function to inversed data"
#     adv_x = round_data(adv_x)
#
#     "convert ndarray to torch.Tensor"
#     adv_x = torch.Tensor(adv_x)
#
#     return adv_x



def get_advX_wf_main(x,pert,pert_box,alpha=None):
    """
    wrap all steps to get adversarial x given website fingerprinting
    alpha: control the size of the perturbation
    x: ndarray
    pert: torch.Tensor
    return: Tensor
    """

    if isinstance(type(x),torch.Tensor):
        x = x.data.cpu().numpy()


    "clamp perturbation"
    pert = torch.clamp(pert, -pert_box, pert_box)
    pert = pert * alpha

    "add pert to x given the restrictions"
    adv_x = add_perturbation(x, pert)

    "only needed for un-normalized data. add round function to inversed data"
    adv_x = round_data(adv_x)

    "convert ndarray to torch.Tensor"
    adv_x = torch.Tensor(adv_x)

    return adv_x