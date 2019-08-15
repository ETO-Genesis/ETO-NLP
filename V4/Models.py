#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : Models.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-13
# Last Modified: 2019-08-15 13:12:50
# Descption    :
# Version      : Python 3.7
############################################
import argparse
from torch import nn
import torch.nn.functional as F
from Net.nn.Encoder.transformer import EncoderTransformer
from Net.Loss.crf import CRFLoss


class TaggerModel(nn.Module):
    def __init__(self, opt, vocab):
        super().__init__()

        opt.encoder['num_embeddings'] = vocab.src_vocab_size
        self.encoder = EncoderTransformer(**opt.encoder)

        self.generator = nn.Sequential(
            nn.Linear(opt.encoder['d_model'], vocab.tgt_vocab_size),
            nn.LogSoftmax(dim=1)
        )
        self.lr = nn.Linear(opt.encoder['d_model'], vocab.tgt_vocab_size)

        opt.tgt_vocab_size = vocab.tgt_vocab_size
        self.criterion = CRFLoss(opt)

    def forward(self, src):
        """
        Args:
            src:
        Returns:
            outputs: (b, s)
        """
        # context: (t, n, e)
        context = self.encoder(src)

        # out = self.generator(context)
        out = self.lr(context)
        out = F.log_softmax(out, dim=1)
        return out


def main(args):
    """main function"""
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
