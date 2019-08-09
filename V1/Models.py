#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : Models.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-13
# Last Modified: 2019-08-09 13:26:35
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import torch.nn as nn
from Net.nn.Encoder.rnn import EncoderRNN
from Net.Loss.crf import CRFLoss


class TaggerModel(nn.Module):
    def __init__(self, opt, vocab):
        super().__init__()

        opt.encoder['vocab_size'] = vocab.src_vocab_size
        self.encoder = EncoderRNN(**opt.encoder)

        self.generator = nn.Sequential(
            nn.Linear(opt.encoder['rnn_size'], vocab.tgt_vocab_size),
            nn.LogSoftmax()
        )

        opt.tgt_vocab_size = vocab.tgt_vocab_size
        self.criterion = CRFLoss(opt)

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


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
