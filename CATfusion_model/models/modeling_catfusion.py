# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
from models.attention import Attention
from models.embed import Embeddings
from models.mlp import Mlp
from models.block import Block
from models.encoder import Encoder
import pdb
from models.model_utils import CoxLoss2

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Transformer(nn.Module):
    def __init__(self, config, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config, vis)

    def forward(self, wsi, omics_inputs_dict=None):

        wsi_embedding, omics_embeddings_dict = self.embeddings(
            wsi, omics_inputs_dict)
        # outputs_dict:{"rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp"}

        text = torch.cat(list(omics_embeddings_dict.values()), 1)
        encoded, attn_weights = self.encoder(wsi_embedding, text)
        return encoded, attn_weights


class CATfusion(nn.Module):
    def __init__(self, config, num_classes=21843, zero_head=False, vis=False):
        super(CATfusion, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, vis)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size*2, 64), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(64, 16), nn.ReLU(), nn.Dropout(p=0.5)
        )

        self.output_range = torch.nn.Parameter(
            torch.FloatTensor([8]), requires_grad=False)
        self.output_shift = torch.nn.Parameter(
            torch.FloatTensor([-4]), requires_grad=False)
        self.head = nn.Sequential(
            nn.Linear(16, num_classes), nn.Sigmoid())

    def forward(self, wsi, omics_inputs_dict=None, survtime=None, censor=None):
        assert len(
            omics_inputs_dict) > 0, "please input at least one omic features"
        x, attn_weights = self.transformer(
            wsi, omics_inputs_dict)
        h = self.mlp(
            torch.cat((x[:, 0, :], torch.mean(x[:, 1:, :], dim=1)), dim=1))
        logits = self.head(h) * self.output_range + self.output_shift
        # print("logits", logits.shape)

        loss_cox = CoxLoss2(survtime, censor, logits, logits.device)
        return loss_cox, logits, x

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            # print(posemb.size(), posemb_new.size())
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" %
                            (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' %
                      (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(
                    np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(
                    gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(
                    gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'CATfusion': configs.get_CATfusion_config(),
}
