import argparse
import os
import sys
from Train import *
import yaml




def main():

    parser = argparse.ArgumentParser()

# #Architecture configs
# #***************************************************************
#     parser.add_argument('--input_img_dim', '-iid', default = "[256,256]", type = str,
#                         help = 'input image dimension, e.g. [256,256]')
#
#     parser.add_argument('--transformer_emb_dim', '-ted', default = 256, type = int,
#                         help = 'transformer d model size')
#
#     parser.add_argument('--transformer_num_heads', '-tnh', default = 2, type = int,
#                         help = 'transformer number of heads')
#
#     parser.add_argument('--transformer_num_encoder_layer', '-tnel', default = 1, type = int,
#                         help = 'transformer number of encoder layer')
#
#     parser.add_argument('--transformer_num_dec_layer', '-tndl', default = 1, type = int,
#                         help = 'transformer number of decoder layer')
#
#     parser.add_argument('--transformer_intermediate_layer_dim', '-tild', default = 512, type = int,
#                         help = 'transformer hidden layers size')
#
#     parser.add_argument('--transformer_dropout_per', '-tdp', default = 0., type = float,
#                         help = 'transformer dropout percentage')
#
#     parser.add_argument('--prior_input_channels', '-pic', default = 3, type = int,
#                         help = 'prior network number of input channels')
#
#     parser.add_argument('--prior_posterior_layers', '-ppl', default = "[64,128,256]", type = str,
#                         help = 'prior network number of input channels')
#
#     parser.add_argument('--posterior_input_channels', '-poic', default = 37, type = int,
#                         help = 'num of image + seg channels')
#
# #***************************************************************
# #Training configs
#     parser.add_argument('--batch_size', '-bs', default = 100, type = int,
#                         help = 'mini batches size')
#     #1e-5
#     parser.add_argument('--momentum', '-mo', default = 1e-4, type = float,
#                         help = 'l2 regularizer')
#
#     parser.add_argument('--epochs', '-e', default = 100, type = int,
#                         help = 'num of epochs')
#
#     parser.add_argument('--learning_rate', '-lr', default = 1e-5, type = float,
#                         help = 'learning rate')
#
#     parser.add_argument('--data_path', '-dp',
#                         default = "../datasets/augmented_cityscapes", type = str,
#                         help = 'path to mapillary dataset. It should be like PATH/dataset')
#
#     parser.add_argument('--run_name', '-rn',
#                         default = "cityscapes", type = str,
#                         help = 'the run name will be apeared in wandb')
#
#
#     parser.add_argument('--project_name', '-pn',
#                         default = "variational_transformer", type = str,
#                         help = 'the run name will be apeared in wandb')
#
#     parser.add_argument('--continue_tra', '-ct',
#                         default = False, type = bool,
#                         help = 'train a model for more epochs. you also need to set the model path.')
#
#
#     parser.add_argument('--wandb_id', '-id',
#                         default = "test", type = str,
#                         help = 'Corresponding wandb run id to resume training.')
#
#
#     args = parser.parse_args()
#     args.input_img_dim = eval(args.input_img_dim)
#     args.prior_posterior_layers = eval(args.prior_posterior_layers)

    #train(**vars(args))

    f = open("config.yml", "r")
    args = yaml.safe_load(f)
    train(**args)




if __name__ == "__main__":
    main()
