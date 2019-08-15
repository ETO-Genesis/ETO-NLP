#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : Models.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-13
# Last Modified: 2019-08-09 13:26:13
# Descption    :
# Version      : Python 3.7
############################################
import torch.nn as nn

from Net.nn.Encoder.rnn import EncoderRNN
from Net.nn.Decoder.rnn import DecoderRNN

from Net.Loss.cross_entropy import NLLLoss


class TaggerModel(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        Inputs: src_seq, input_lengths, tgt_seq
        Outputs:
            outputs: 概率矩阵
    """

    def __init__(self, opt, vocab):
        super().__init__()

        opt.encoder['vocab_size'] = vocab.src_vocab_size
        self.encoder = EncoderRNN(**opt.encoder)
        opt.decoder['vocab_size'] = vocab.tgt_vocab_size
        opt.decoder['device'] = opt.device
        self.decoder = DecoderRNN(**opt.decoder)

        self.criterion = NLLLoss(opt)

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, src_seq, input_lengths=None, tgt_seq=None):
        enc_outputs, enc_hidden = self.encoder(src_seq, input_lengths)

        if tgt_seq is None:
            self.decoder.max_lenth = src_seq.size(1)
            tgt_seq = src_seq

        # 注意：Encoder可能是双向的，而Decoder是单向的，此从下往上取n_layers个
        br = 2 if self.decoder.brnn else 1
        enc_hidden = enc_hidden[:self.decoder.layers * br]

        result = self.decoder(
            inputs=tgt_seq,
            enc_hidden=enc_hidden,
            enc_outputs=enc_outputs
        )
        return result
