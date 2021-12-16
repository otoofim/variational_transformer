
import functools
from collections import OrderedDict
import torch
import torch.nn as nn
import operator
from UNet import *
from torch.distributions import Normal, Independent, kl



class AxisAlignedConvGaussian(nn.Module):

    def __init__(self, input_channels, filters_enc, inp_dim):

        super(AxisAlignedConvGaussian, self).__init__()

        self.input_channels = input_channels
        self.filters_enc = filters_enc[:-1]
        self.latent_dim = filters_enc[-1]
        self.inp_dim = inp_dim

        #Encoder
        layers = {}
        output_dim = 0
        for i, filter in enumerate(self.filters_enc):

            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = filter

            layers["enc{}_unetblock".format(i)] = UNet._block(input_dim, output_dim, "enc{}".format(i))
            layers["enc{}_avgpooling".format(i)] = nn.AvgPool2d(kernel_size = 2, stride = 2)

        num_features_before_fcnn = functools.reduce(operator.mul, list(nn.Sequential(OrderedDict(layers))(torch.rand(1, *(self.input_channels, self.inp_dim[0], self.inp_dim[1]))).shape))

        layers["linear_layer"] = nn.Linear(num_features_before_fcnn, self.latent_dim * 2)
        self.enc_layers = nn.ModuleDict(layers)


    def forward(self, input, segm = None):

        enc_outputs = None
        output = input

        if segm is not None:
            output = torch.cat((input, segm), dim=1)


        for layer in self.enc_layers:
            if layer == "linear_layer":
                feature = functools.reduce(operator.mul, output.shape[1:], 1)
                output = self.enc_layers[layer](output.view(-1, feature))
            else:
                output = self.enc_layers[layer](output)

        mu = output[:,:self.latent_dim]
        log_sigma = output[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)

        return dist
