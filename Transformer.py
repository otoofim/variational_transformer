# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model = 512, nhead = 8, num_encoder_layers = 6,
                 num_decoder_layers = 6, dim_feedforward = 2048, dropout = 0,
                 activation = "relu", return_intermediate_dec = False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                          return_intermediate = return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, embeded_input, embeded_latent, selfAttn_mask = None, selfAttn_padding_mask = None):

        encoder_output = self.encoder(embeded_input, selfAttn_mask = selfAttn_mask, selfAttn_padding_mask = selfAttn_padding_mask)
        decoder_output = self.decoder(embeded_latent, encoder_output, EncDecAttn_key_padding_mask = selfAttn_padding_mask)

        return decoder_output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):

        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, embeded_input, selfAttn_mask = None,  selfAttn_padding_mask = None):

        output = embeded_input
        for layer in self.layers:
            output = layer(output, selfAttn_mask, selfAttn_padding_mask)
            #output = layer(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate = False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, embeded_latent, encoder_output,
                     selfAttn_mask = None,
                     EncDecAttn_mask = None,
                     selfAttn_key_padding_mask = None,
                     EncDecAttn_key_padding_mask = None):

        output = embeded_latent
        intermediate = []

        for layer in self.layers:
            output = layer(output, encoder_output,
                             selfAttn_mask,
                             EncDecAttn_mask,
                             selfAttn_key_padding_mask,
                             EncDecAttn_key_padding_mask)

            if self.return_intermediate:
                intermediate.append(self.norm(output))


        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0,
             activation="relu"):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout, batch_first = True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def forward(self,
                 embeded_input,
                 selfAttn_mask = None,
                 selfAttn_padding_mask = None):

        attn = self.self_attn(query = embeded_input, key = torch.transpose(embeded_input, 1, 2), value = embeded_input, key_padding_mask = selfAttn_padding_mask, attn_mask = selfAttn_mask)[0]

        attn = embeded_input + self.dropout1(attn)
        attn = self.norm1(attn)
        linear_output = self.linear2(self.dropout(self.activation(self.linear1(attn))))
        linear_output = attn + self.dropout2(linear_output)
        linear_output = self.norm2(linear_output)
        return linear_output




class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0,
                 activation="relu"):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout, batch_first = True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout, batch_first = True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def forward(self, embeded_latent, encoder_output,
                     selfAttn_mask = None,
                     EncDecAttn_mask = None,
                     selfAttn_key_padding_mask = None,
                     EncDecAttn_key_padding_mask = None):


        quary_att = self.self_attn(query = embeded_latent, key = embeded_latent,
                            value = embeded_latent, key_padding_mask = selfAttn_key_padding_mask, attn_mask = selfAttn_mask)[0]


        quary_att = embeded_latent + self.dropout1(quary_att)
        quary_att = self.norm1(quary_att)

        multi_attn = self.multihead_attn(query = quary_att, key = encoder_output,
                            value = encoder_output, key_padding_mask = EncDecAttn_key_padding_mask, attn_mask = EncDecAttn_mask)[0]


        multi_attn = quary_att + self.dropout2(multi_attn)
        multi_attn = self.norm2(multi_attn)
        linear_output = self.linear2(self.dropout(self.activation(self.linear1(multi_attn))))
        linear_output = multi_attn + self.dropout3(linear_output)
        linear_output = self.norm3(linear_output)
        return linear_output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
