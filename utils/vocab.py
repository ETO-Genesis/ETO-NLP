#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : vocab.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-03-07
# Last Modified: 2019-08-15 11:16:08
# Descption    : Handling the data io
# Version      : Python 3.7
############################################
import argparse
import time
import os
import torch
from tqdm import tqdm

from Net import Constants
import logging


class Vocab(object):
    def __init__(self, opt):

        src_word2idx, tgt_word2idx = {}, {}
        if opt.vocab and os.path.exists(opt.vocab):
            vocab = torch.load(opt.vocab)

            print('[Info] Pre-defined vocabulary found.')
            src_word2idx = vocab['src']
            tgt_word2idx = vocab['tgt']

        self._src_word2idx = src_word2idx
        self._tgt_word2idx = tgt_word2idx

        self._src_idx2word = {idx: word for word, idx in src_word2idx.items()}
        self._tgt_idx2word = {idx: word for word, idx in tgt_word2idx.items()}

    @property
    def src_word2idx(self):
        return self._src_word2idx

    @src_word2idx.setter
    def src_word2idx(self, src_word2idx):
        self.src_word2idx = src_word2idx

    @property
    def src_idx2word(self):
        return self._src_idx2word

    @property
    def tgt_word2idx(self):
        return self._tgt_word2idx

    @property
    def tgt_idx2word(self):
        return self._tgt_idx2word

    @property
    def src_vocab_size(self):
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        return len(self._tgt_word2idx)

    def save(self, vocab_path):
        vocab = {
            "src": self._src_word2idx,
            "tgt": self._tgt_word2idx
        }
        logging.INFO(f"Vocab save {opt.vocab} ...")
        torch.save(vocab, vocab_path)


def main(opt):
    return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', default=False)

    opt = parser.parse_args()
    main(opt)
