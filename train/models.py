import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.autograd as autograd


#----------------------------------------------
class cnn_noNorm(nn.Module):
    "works for the data that already normalized"

    def __init__(self,params):
        self.params = params
        super(cnn_noNorm, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv1_input_channel'], out_channels=self.params['conv1_output_channel'],
                      kernel_size=self.params['kernel_size1'], stride=self.params['stride1'], padding=self.params['padding1']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate1']),
            nn.MaxPool1d(kernel_size=self.params['pool1'])
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv2_input_channel'], out_channels=self.params['conv2_output_channel'],
                      kernel_size=self.params['kernel_size2'], stride=self.params['stride2'],
                      padding=self.params['padding2']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate2']),
            nn.MaxPool1d(kernel_size=self.params['pool2'])
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv3_input_channel'], out_channels=self.params['conv3_output_channel'],
                      kernel_size=self.params['kernel_size3'], stride=self.params['stride3'],
                      padding=self.params['padding3']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate3']),
            nn.MaxPool1d(kernel_size=self.params['pool3'])
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv4_input_channel'], out_channels=self.params['conv4_output_channel'],
                      kernel_size=self.params['kernel_size4'], stride=self.params['stride4'],
                      padding=self.params['padding4']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate4']),
            nn.MaxPool1d(kernel_size=self.params['pool4'])
        )

        #add flatten layer, the output dimension of flatten is the input of FC num_layers
        # self.flatten = nn.Flatten()
        # self.out_para1 = self.flatten.shape[1]
        self.out_param1 = math.ceil(math.ceil(math.ceil(math.ceil(self.params['input_size']/self.params['pool1'])/self.params['pool2'])/self.params['pool3'])/self.params['pool4'])
        self.out = nn.Linear(self.params['conv4_output_channel']*self.out_param1,self.params['num_classes'])


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        logits = self.out(x)
        return logits



#----------------------------------------------
class cnn_norm(nn.Module):
    "works for the website fingerprinting burst data without normalized"

    def __init__(self,params):
        self.params = params
        super(cnn_norm, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv1_input_channel'], out_channels=self.params['conv1_output_channel'],
                      kernel_size=self.params['kernel_size1'], stride=self.params['stride1'], padding=self.params['padding1']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate1']),
            nn.BatchNorm1d(self.params['conv1_output_channel']),
            nn.MaxPool1d(kernel_size=self.params['pool1'])
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv2_input_channel'], out_channels=self.params['conv2_output_channel'],
                      kernel_size=self.params['kernel_size2'], stride=self.params['stride2'],
                      padding=self.params['padding2']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate2']),
            nn.BatchNorm1d(self.params['conv2_output_channel']),
            nn.MaxPool1d(kernel_size=self.params['pool2'])
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv3_input_channel'], out_channels=self.params['conv3_output_channel'],
                      kernel_size=self.params['kernel_size3'], stride=self.params['stride3'],
                      padding=self.params['padding3']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate3']),
            nn.BatchNorm1d(self.params['conv3_output_channel']),
            nn.MaxPool1d(kernel_size=self.params['pool3'])
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv4_input_channel'], out_channels=self.params['conv4_output_channel'],
                      kernel_size=self.params['kernel_size4'], stride=self.params['stride4'],
                      padding=self.params['padding4']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate4']),
            nn.BatchNorm1d(self.params['conv4_output_channel']),
            nn.MaxPool1d(kernel_size=self.params['pool4'])
        )

        self.out_param1 = math.ceil(math.ceil(math.ceil(math.ceil(self.params['input_size']/self.params['pool1'])/self.params['pool2'])/self.params['pool3'])/self.params['pool4'])
        self.out = nn.Linear(self.params['conv4_output_channel']*self.out_param1,self.params['num_classes'])


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        logits = self.out(x)
        return logits




#----------------------------------------------
class cnn(nn.Module):
    "simple structure, works for the data that already normalized"

    def __init__(self,params):
        self.params = params
        super(cnn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv1_input_channel'], out_channels=self.params['conv1_output_channel'],
                      kernel_size=self.params['kernel_size1'], stride=self.params['stride1'], padding=self.params['padding1']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate1']),
            nn.MaxPool1d(kernel_size=self.params['pool1'])
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv2_input_channel'], out_channels=self.params['conv2_output_channel'],
                      kernel_size=self.params['kernel_size2'], stride=self.params['stride2'],
                      padding=self.params['padding2']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate2']),
            nn.MaxPool1d(kernel_size=self.params['pool2'])
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.params['conv3_input_channel'], out_channels=self.params['conv3_output_channel'],
        #               kernel_size=self.params['kernel_size3'], stride=self.params['stride3'],
        #               padding=self.params['padding3']),
        #     nn.ReLU(),
        #     nn.Dropout(self.params['drop_rate3']),
        #     nn.MaxPool1d(kernel_size=self.params['pool3'])
        # )

        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.params['conv4_input_channel'], out_channels=self.params['conv4_output_channel'],
        #               kernel_size=self.params['kernel_size4'], stride=self.params['stride4'],
        #               padding=self.params['padding4']),
        #     nn.ReLU(),
        #     nn.Dropout(self.params['drop_rate4']),
        #     nn.MaxPool1d(kernel_size=self.params['pool4'])
        # )

        self.out_param1 = math.ceil(math.ceil(math.ceil(math.ceil(self.params['input_size']/self.params['pool1'])/self.params['pool2'])))
        self.out = nn.Linear(self.params['conv2_output_channel']*self.out_param1,self.params['num_classes'])


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = x.view(x.size(0),-1)
        logits = self.out(x)
        return logits




#----------------------------------------------
class fcnn(nn.Module):
    def __init__(self,opts):

        super(fcnn,self).__init__()

        self.opts = opts
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Linear(256, self.opts['num_classes']),
        )

    def forward(self, input):
        ouput_flat = input.view(input.shape[0], -1)
        logits = self.model(ouput_flat)
        return logits






#----------------------------------------------
class lstm(nn.Module):
    def __init__(self,opts):

        super(lstm,self).__init__()

        self.opts = opts
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

        "design lstm"
        self.lstm = nn.LSTM(input_size=self.opts['input_size'],hidden_size=self.opts['hidden_dim'],num_layers=self.opts['num_layers'],
                            dropout=self.opts['dropout'],batch_first=self.opts['batch_first'],bidirectional=self.opts['bidirectional'])

        "dense layer: output layer with logits"
        self.hidden2out = nn.Linear(self.opts['hidden_dim'],self.opts['num_classes'])

        "initial hidden state"
        # self.hidden = self.init_hidden()

        "softmax"
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        return (torch.zeros(self.opts['num_layers'], self.opts['batch_size'], self.opts['hidden_dim']).to(self.device),
                torch.zeros(self.opts['num_layers'], self.opts['batch_size'], self.opts['hidden_dim']).to(self.device))


    def act_fun(self,x):
        if self.opts['activation_fun'] == 'relu':
            activate_f = F.relu(x)
        elif self.opts['activation_fun'] == 'tanh':
            activate_f = F.tanh(x)
        elif self.opts['activation_fun'] == 'elu':
            activate_f = F.elu(x)
        elif self.opts['activation_fun'] == 'selu':
            activate_f = F.selu(x)
        elif self.opts['activation_fun'] == 'leaky_relu':
            activate_f = F.leaky_relu(x)
        elif self.opts['activation_fun'] == 'sigmoid':
            activate_f = F.sigmoid(x)

        return activate_f


    def forward(self,x):
        """
        Forward pass through LSTM layer
        shape of lstm_out: [input_size, batch_size, hidden_dim]
        shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        """

        "aviod error: rnn model chunk of memory"
        self.lstm.flatten_parameters()

        lstm_out, (h_n,h_c) = self.lstm(x,None)   # hidden state is none, default is zero
        out = lstm_out.view(len(x),-1)

        # out = F.relu(out)
        out = self.act_fun(out)

        out = self.hidden2out(out)
        # logits = self.softmax(out)
        # logits = torch.log_softmax(out,dim=1)
        logits = out

        return logits




#----------------------------------------------
class rnn(nn.Module):
    def __init__(self,opts):

        super(rnn,self).__init__()

        self.opts = opts
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

        "design lstm"
        self.rnn = nn.RNN(input_size=self.opts['input_size'],hidden_size=self.opts['hidden_dim'],num_layers=self.opts['num_layers'],
                            dropout=self.opts['dropout'],batch_first=self.opts['batch_first'],bidirectional=self.opts['bidirectional'])

        "dense layer: output layer with logits"
        self.hidden2out = nn.Linear(self.opts['hidden_dim'],self.opts['num_classes'])

        "initial hidden state"
        # self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.opts['num_layers'], self.opts['batch_size'], self.opts['hidden_dim']).to(self.device),
                torch.zeros(self.opts['num_layers'], self.opts['batch_size'], self.opts['hidden_dim']).to(self.device))


    def act_fun(self,x):
        if self.opts['activation_fun'] == 'relu':
            activate_f = F.relu(x)
        elif self.opts['activation_fun'] == 'tanh':
            activate_f = F.tanh(x)
        elif self.opts['activation_fun'] == 'elu':
            activate_f = F.elu(x)
        elif self.opts['activation_fun'] == 'selu':
            activate_f = F.selu(x)
        elif self.opts['activation_fun'] == 'leaky_relu':
            activate_f = F.leaky_relu(x)
        elif self.opts['activation_fun'] == 'sigmoid':
            activate_f = F.sigmoid(x)

        return activate_f


    def forward(self,x):
        """
        Forward pass through rnn layer
        shape of rnn_out: [input_size, batch_size, hidden_dim]
        shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        """

        "aviod error: rnn model chunk of memory"
        self.rnn.flatten_parameters()

        lstm_out, hidden_out = self.rnn(x,None)   # hidden state is none, default is zero
        out = lstm_out.view(len(x),-1)

        # out = F.relu(out)
        out = self.act_fun(out)

        out = self.hidden2out(out)
        logits = torch.log_softmax(out,dim=1)

        return logits





#----------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv1d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output

#----------------------------------------------
class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv1d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm1d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm1d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose1d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


#----------------------------------------------
# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm1d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
