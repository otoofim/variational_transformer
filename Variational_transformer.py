import torch
import torch.nn as nn
from Transformer import *
from VAE import *


class VariationalTransformer(nn.Module):

    def __init__(self, **kwargs):

        super(VariationalTransformer, self).__init__()

        
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)


        self.transformer = Transformer(d_model = kwargs['transformer_emb_dim'], nhead = kwargs['transformer_num_heads'],
                                        num_encoder_layers = kwargs['transformer_num_encoder_layer'], num_decoder_layers = kwargs['transformer_num_dec_layer'],
                                        dim_feedforward = kwargs['transformer_intermediate_layer_dim'], dropout = kwargs['transformer_dropout_per'],
                                        activation = "relu", normalize_before = False, return_intermediate_dec = False)

        self.decoder_emb = nn.Linear(kwargs['prior_posterior_layers'][-1], kwargs['transformer_emb_dim'])

        self.prior = AxisAlignedConvGaussian(input_channels = kwargs['prior_input_channels'], filters_enc = kwargs['prior_posterior_layers'], inp_dim = kwargs['input_img_dim'])
        self.posterior = AxisAlignedConvGaussian(input_channels = kwargs['posterior_input_channels'], filters_enc = kwargs['prior_posterior_layers'], inp_dim = kwargs['input_img_dim'])

    def inference(self, img):

        prior_latent_space = self.prior.forward(img)
        transformer_encoder_output = self.transformer.encoder.forward(self.backbone(img))
        for _ in range(16):
            latent_vector_prior = self.sample(prior_latent_space, training = False)
            #?????????????????????????????????????????????????????????????????????????
            reconstruct_prior = self.transformer.decoder.forward(transformer_encoder_output, self.decoder_emb(latent_vector_prior))
            yield reconstruct_prior


    def forward(self, img, segm):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        prior_latent_space = self.prior.forward(img)
        latent_vector_prior = self.sample(prior_latent_space, True)
        reconstruct_prior = self.transformer.forward(self.backbone(img), self.decoder_emb(latent_vector_prior))

        posterior_latent_space = self.posterior.forward(img, segm)
        latent_vector_posterior = self.sample(posterior_latent_space, training)
        reconstruct_posterior = self.transformer.forward(self.backbone(img), self.decoder_emb(latent_vector_posterior))
        return prior_latent_space, posterior_latent_space, reconstruct_posterior, reconstruct_prior


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
