import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data


def params_cnn(num_class,input_size):
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
        'input_size': input_size
    }


def params_lstm(num_class,input_size,batch_size):
    "hyperparameters for target model"

    print('input_size: {}'.format(input_size))

    return {
        'num_layers':2,
        'hidden_dim':128,
        'dropout': 0.1,
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
        'num_layers':2,
        'hidden_dim':128,
        'dropout': 0,
        'batch_size': batch_size,
        'batch_first':True,
        'bidirectional':False,
        'num_classes': num_class,
        'input_size':input_size
    }



def incoming_keeper(x):
    """
    for given adversarial network traffic,
    keep each paket negative as it is a incoming packet,
    :param x:
    :return:
    """
    result = []

    for p in x:
        if p >= -1 and p <0:
            result.append(p)
        # elif p > 1:
        #     p = 1
        elif p < 0:
            p = 0
            result.append(p)

    return result


def incoming_batch_keeper(x):
    """

    :param x: multi-traffic in batch in numpy.array format instead tensor, [batch_size,len_traffic]
    :return:
    """

    # squeeze: remove all 1 dimension
    # [batch, 1, len_taffic] to [batch, len_traffic]
    x = np.squeeze(x)

    group = []
    if len(x.shape) > 1:
        for i in range(len(x)):
            row_i = x[i,:]
            row_i = incoming_keeper(row_i)
            group.append(row_i)
    else:
        x = incoming_keeper(x)
        group.append(x)
    group = np.array(group)

    # add dimension
    group = np.expand_dims(group,axis=1)
    return group


def ndarray_tensor(x):
    """
    ndarray to tensor
    :param x:
    :return:
    """
    x = torch.Tensor(x)
    return x


def tensor_ndarray(x):
    x = x.data.cpu().numpy()
    return x


def traffic_plot(fig_id,x,x_adv):

    def plot(i):
        plt.subplot(2,2,i)
        plt.plot(x[i-1] * 1500,label='network traffic')
        plt.plot(x_adv[i-1] * 1500,'--',label='adv network traffic')
        # plt.xticks(np.arange(0,300,40))
        # plt.plot(noise[i-1],label='perturbation')
        plt.legend()

    for i in range(1,5):
        plot(i)

    plt.savefig('./fig/fig_'+str(fig_id)+'.eps')
    plt.show()


def single_traffic_plot(fig_id,x,x_adv,Adversary):

    fontsize = 18
    ticks_size = 15

    plt.figure(figsize=(8, 6), dpi=150)
    # plt.title('Network Traffics: ' + Adversary)
    plt.plot(x,linewidth=2,label='Original trace',)
    plt.plot(x_adv,linestyle='--',linewidth=2,label='Aversarial trace')
    plt.xlabel('Trace Length', {'size': fontsize})
    plt.ylabel('Burst Size', {'size': fontsize})
    plt.legend(loc='best',fontsize=fontsize)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)

    plt.savefig('../fig/' + Adversary + '_' + 'traffic_' + str(fig_id)+'.eps')
    # plt.show()


def noise_plot(fig_id,noise,Adversary):

    plt.figure()
    plt.title('Perturbation: ' + Adversary)
    plt.plot(noise, label='perturbation')
    # plt.ylim(-200,200)
    plt.legend()
    plt.savefig('../fig/' + Adversary + '_' + 'pert_' + str(fig_id)+'.eps')
    # plt.show()


def load_data(path):
    "col_length: column length of each input"

    raw_data = pd.read_csv(path)
    x = raw_data.iloc[:,:-1]
    label = raw_data['label']

    return x, label


def data_preprocess(x, y,batch_size):
    """
    input: X, y ( X: dataframe, y: ndarray)
    output: tensor_loader. train_loader, test_loader
    """
    "normalize X"
    x = x / 1500.0

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
        shuffle = False,
    )

    return data_loader


def load_data_main(path,batch_size):
    x,y = load_data(path)
    data_loader = data_preprocess(x,y,batch_size)
    return data_loader



# def add_perturbation(x,pert):
#     """
#     the perturbation add to burst format data in Website Finerprinting
#     must be keep the result increase in the original direction
#     e.g., x = [-2,3,-1], pert = [-0.03,-0.04,-0.1], x+pert=[-2-0.03,3+0,-1-0.1]=[-2.03,3,-1.1]
#
#     :param x: ndarray
#     :param pert: ndarray
#     :return: torch.Tensor
#     """
#
#     input_shape = x.shape
#
#     "Tensor to ndarray"
#     if type(x) == torch.Tensor:
#         x = x.data.cpu().numpy()
#     if type(pert) == torch.Tensor:
#         pert = pert.data.cpu().numpy()
#
#
#     result = []
#
#     for i in range(len(x)):
#         for j in range(len(x[i][0])):
#             a = x[i][0][j]
#             b = pert[i][0][j]
#             if a * b >= 0:
#                 temp = a + b
#                 result.append(temp)
#             else:
#                 temp = a
#                 result.append(temp)
#
#     result = torch.Tensor(np.array(result))
#     result = result.view(input_shape)
#
#     return result
