import torch
import torch.nn as nn
from Transformer import *
from PP import *
import math
import functools
import operator

def prior_last_layer(dim_in, stride = [1, 1], padding = [0, 0], dilation = [1, 1], kernel_size = [1, 1], output_padding = [0, 0]):

    return ((dim_in + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) /  stride[0]) + 1


def choose_backbone():

    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    backbone = torch.nn.Sequential(*(list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).children())[:7]))
    backbone.requires_grad = False
    return backbone


class VariationalTransformer(nn.Module):

    def __init__(self, **kwargs):

        super(VariationalTransformer, self).__init__()

        self.batch_size = kwargs["batch_size"]
        self.num_samples = kwargs["num_samples"]
        self.backbone = choose_backbone()

        self.backbone_output_dim = functools.reduce(operator.mul, self.backbone(torch.rand(1, *(kwargs['prior_input_channels'], kwargs['input_img_dim'][0], kwargs['input_img_dim'][1])))).shape
        self.seq_length = self.backbone_output_dim[0]

        dim1 = prior_last_layer(self.backbone_output_dim[1])
        dim2 = prior_last_layer(self.backbone_output_dim[2])
        last_layer = int(dim1 * dim2)
        layers = list(kwargs['prior_posterior_layers'])
        layers.append(last_layer)


        self.transformer = Transformer(d_model = last_layer, nhead = kwargs['transformer_num_heads'],
                                        num_encoder_layers = kwargs['transformer_num_encoder_layer'], num_decoder_layers = kwargs['transformer_num_dec_layer'],
                                        dim_feedforward = kwargs['transformer_intermediate_layer_dim'], dropout = kwargs['transformer_dropout_per'],
                                        activation = "relu", return_intermediate_dec = False)

        self.decoder_emb = nn.ConvTranspose2d(1, self.seq_length, kernel_size = 1, stride = 1)

#         self.prior = AxisAlignedConvGaussian(input_channels = kwargs['prior_input_channels'], filters_enc = layers, inp_dim = kwargs['input_img_dim'])
#         self.posterior = AxisAlignedConvGaussian(input_channels = kwargs['posterior_input_channels'], filters_enc = layers, inp_dim = kwargs['input_img_dim'])

        self.prior = AxisAlignedConvGaussian(input_channels = kwargs['prior_input_channels'], num_filters = layers, no_convs_per_block = kwargs['pp_cnn_per_block'], latent_dim = kwargs['latent_dim']).to(device)
        self.posterior = AxisAlignedConvGaussian(input_channels = kwargs['posterior_input_channels'], num_filters = layers, no_convs_per_block = kwargs['pp_cnn_per_block'], latent_dim = kwargs['latent_dim'], posterior=True).to(device)

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = kwargs["num_cat"], kernel_size = 3, padding = 1, bias = True),
            nn.Softmax(dim=1)
        )

    def inference(self, img):

        prior_latent_space = self.prior.forward(img)
        resnet_features = self.backbone(img)
        transformer_encoder_output = self.transformer.encoder.forward(resnet_features.contiguous().view(img.shape[0], self.seq_length, -1))
        for _ in range(self.num_samples):
            latent_vector_prior = self.sample(prior_latent_space, training = False)
            decoder_embedding = self.decoder_emb(latent_vector_prior.unsqueeze(1).view(img.shape[0], 1, int(math.sqrt(latent_vector_prior.shape[1])), -1))
            reconstruct_prior = self.transformer.decoder.forward(transformer_encoder_output, decoder_embedding.contiguous().view(img.shape[0], self.seq_length, -1))
            reconstruct_prior = self.output_layer(reconstruct_prior.unsqueeze(1))
            yield reconstruct_prior


    def forward(self, img, segm):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        prior_latent_space = self.prior.forward(img)
        latent_vector_prior = self.sample(prior_latent_space, True)
        #reconstruct_prior = self.transformer.forward(self.backbone(img), self.decoder_emb(latent_vector_prior))

        posterior_latent_space = self.posterior.forward(img, segm)
        latent_vector_posterior = self.sample(posterior_latent_space, True)


        resnet_features = self.backbone(img)
        decoder_embedding = self.decoder_emb(latent_vector_posterior.unsqueeze(1).view(self.batch_size, 1, int(math.sqrt(latent_vector_posterior.shape[1])), -1))
        reconstruct_posterior = self.transformer.forward(resnet_features.contiguous().view(self.batch_size, self.seq_length, -1), decoder_embedding.contiguous().view(self.batch_size, self.seq_length, -1))
        reconstruct_posterior = self.output_layer(reconstruct_posterior.unsqueeze(1))

        return prior_latent_space, posterior_latent_space, reconstruct_posterior


    def sample(self, dist, training = False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if training == True:
            z_prior = dist.rsample()
        else:
            z_prior = dist.sample()

        return z_prior
