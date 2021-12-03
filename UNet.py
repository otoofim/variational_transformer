
import functools
from collections import OrderedDict
import torch
import torch.nn as nn
import operator

class UNet(nn.Module):
    def __init__(self, input_channels, filters_enc, output_channels):

        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters_enc = filters_enc
        self.filters_dec = filters_enc[::-1]
        self.bottelneck = filters_enc[-1] * 2


        #Encoder
        layers = {}
        output_dim = 0
        for i, filter in enumerate(self.filters_enc):

            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = filter

            layers["enc{}_unetblock".format(i)] = UNet._block(input_dim, output_dim, "enc{}".format(i))
            layers["enc{}_maxpooling".format(i)] = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.enc_layers = nn.ModuleDict(layers)

        #Bottleneck
        self.bottelneck_layers = nn.Sequential(
            UNet._block(self.filters_enc[-1], self.bottelneck, name = "bottleneck"),
            nn.ConvTranspose2d(self.bottelneck, self.filters_enc[-1], kernel_size = 2, stride = 2)
        )

        #Decoder
        layers = []
        output_dim = 0
        for i, filter in enumerate(self.filters_dec):

            input_dim = self.filters_enc[-1] * 2 if i == 0 else output_dim * 2
            output_dim = int(filter/2)

            layers.append(nn.Sequential(
                UNet._block(input_dim, output_dim, "dec{}".format(i)),
                nn.ConvTranspose2d(output_dim, output_dim, kernel_size = 2, stride = 2)
            ))

        self.dec_layers = nn.ModuleList(layers)

        self.final_layer = nn.Sequential(
            UNet._block(output_dim, self.output_channels, "final_layer")
        )


        #self.layers.apply(init_weights)


    def forward(self, x):

        enc_outputs = {}
        output = x
        for layer in self.enc_layers:
            output = self.enc_layers[layer](output)
            enc_outputs[layer] = output

        bott_output = self.bottelneck_layers(enc_outputs["enc{}_maxpooling".format(len(self.filters_enc) - 1)])
        dec_output = bott_output

        for i, layer in enumerate(self.dec_layers):
            key = "enc{}_unetblock".format(len(self.filters_enc) -1 -i)
            dec_output = layer(torch.cat((dec_output, enc_outputs[key]), dim=1))

        final_out = self.final_layer(dec_output)

        return final_out



    @staticmethod
    def _block(in_channels, features_out, name, mid_feature = 0, midle_layers = 0):

        layers = {}
        layers["{}_conv1".format(name)] = UNet._unit_block(in_channels = in_channels, out_channels = (features_out if midle_layers == 0 else mid_feature), name = "{}_conv1".format(name))

        for i in range(midle_layers):
            layers["{}_conv{}".format(name, i + 2)] = UNet._unit_block(in_channels = mid_feature, out_channels = mid_feature, name = "{}_conv{}".format(name, i + 2))

        layers["{}_conv{}".format(name, midle_layers + 2)] = UNet._unit_block(in_channels = (features_out if midle_layers == 0 else mid_feature), out_channels = features_out, name = "{}_conv{}".format(name, midle_layers + 2))

        return nn.Sequential(OrderedDict(layers))


    @staticmethod
    def _unit_block(in_channels, out_channels, name):

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "_norm", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu", nn.ReLU(inplace=True)),
                ]
            )
        )
