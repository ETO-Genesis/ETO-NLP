#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : Models.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-13
# Last Modified: 2019-08-08 16:19:37
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import torch
import torch.nn as nn
from Net.nn.Encoder.rnn import EncoderRNN


class TaggerModel(nn.Module):
    def __init__(self, opt, vocab):
        super().__init__()

        self.encoder = EncoderRNN(opt.encoder)

        self.generator = nn.Sequential(
            nn.Linear(opt.rnn_size, vocab.tgt_vocab_size),
            nn.LogSoftmax()
        )

    def forward(self, src):
        """
        Args:
            src:
        Returns:
            outputs: (b, s)
        """
        src = src
        context, hidden = self.encoder(src)
        out = self.generator(context)
        return out
