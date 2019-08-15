#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : Translator.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-03-07
# Last Modified: 2019-08-12 11:07:21
# Descption    : ''' This module will handle the text generation with beam search. '''
# Version      : Python 3.7
############################################
import argparse
import torch

from models import Models


class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt, vocab, device):
        self.opt = opt

        self.device = device
        print(self.device)

        checkpoint = torch.load(opt.model)

        model = Models.TaggerModel(
            opt,
            vocab
        )
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        self.model = model.to(self.device)
        self.model.eval()

    def translate(self, src_seq):
        src_seq = src_seq.to(self.device)
        with torch.no_grad():
            batch_hyp = self.model(src_seq)
            batch_hyp = torch.argmax(batch_hyp, dim=-1)
        batch_hyp = torch.cat((src_seq[:, 0].unsqueeze(1), batch_hyp), dim=1)
        return batch_hyp
