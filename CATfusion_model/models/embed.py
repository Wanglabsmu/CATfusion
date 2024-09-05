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
import pdb


class Embeddings(nn.Module):
    """Construct the embeddings from wsi, omics, position embeddings.
    """

    def __init__(self, config):
        super(Embeddings, self).__init__()
        # "rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp"

        self.patch_embeddings = Linear(config.wsi_dim, config.hidden_size)
        self.rnaseq_embeddings = Linear(config.omics_dim, config.hidden_size)
        self.mirnaseq_embeddings = Linear(config.omics_dim, config.hidden_size)
        self.copynumbervariation_embeddings = Linear(
            config.omics_dim, config.hidden_size)
        self.dnamethylationvariation_embeddings = Linear(
            config.omics_dim, config.hidden_size)
        self.snp_embeddings = Linear(config.omics_dim, config.hidden_size)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, 1+config.wsi_feature_len, config.hidden_size))
        self.pe_rnaseq = nn.Parameter(
            torch.zeros(1, config.omics_feature_len.rna_seq, config.hidden_size))
        self.pe_mirnaseq = nn.Parameter(
            torch.zeros(1, config.omics_feature_len.mirna_seq, config.hidden_size))
        self.pe_copynumbervariation = nn.Parameter(
            torch.zeros(1, config.omics_feature_len.copynumbervariation, config.hidden_size))
        self.pe_dnamethylationvariation = nn.Parameter(
            torch.zeros(1, config.omics_feature_len.dnamethylationvariation, config.hidden_size))
        self.pe_snp = nn.Parameter(torch.zeros(
            1, config.omics_feature_len.snp, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_rnaseq = Dropout(config.transformer["dropout_rate"])
        self.dropout_mirnaseq = Dropout(config.transformer["dropout_rate"])
        self.dropout_copynumbervariation = Dropout(
            config.transformer["dropout_rate"])
        self.dropout_dnamethylationvariation = Dropout(
            config.transformer["dropout_rate"])
        self.dropout_snp = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, omics_inputs_dict):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        omics_embeddings_fn = {
            "rna_seq": self.rnaseq_embeddings,
            "mirna_seq": self.mirnaseq_embeddings,
            "copynumbervariation": self.copynumbervariation_embeddings,
            "dnamethylationvariation": self.dnamethylationvariation_embeddings,
            "snp": self.snp_embeddings
        }

        omics_embeddings_pe = {
            "rna_seq": self.pe_rnaseq,
            "mirna_seq": self.pe_mirnaseq,
            "copynumbervariation": self.pe_copynumbervariation,
            "dnamethylationvariation": self.pe_dnamethylationvariation,
            "snp": self.pe_snp
        }

        omics_embeddings_dropout = {
            "rna_seq": self.dropout_rnaseq,
            "mirna_seq": self.dropout_mirnaseq,
            "copynumbervariation": self.dropout_copynumbervariation,
            "dnamethylationvariation": self.dropout_dnamethylationvariation,
            "snp": self.dropout_snp
        }
        x = self.patch_embeddings(x)  # 768=>768
        omics_embeddings = dict()
        for omic in omics_inputs_dict:
            tmp_omic_embeddings = omics_embeddings_fn[omic](
                omics_inputs_dict[omic])
            tmp_omic_embeddings = tmp_omic_embeddings + \
                omics_embeddings_pe[omic]
            tmp_omic_embeddings = omics_embeddings_dropout[omic](
                tmp_omic_embeddings)
            omics_embeddings[omic] = tmp_omic_embeddings

        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings, omics_embeddings
