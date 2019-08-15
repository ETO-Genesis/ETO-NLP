#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : train.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-03-07
# Last Modified: 2019-08-15 13:15:03
# Descption    : This script handling the training process.
# Version      : Python 3.7
############################################
import argparse
import os
import time
import math
from tqdm import tqdm

import torch
from torch import optim
import torch.utils.data

from models.Models import TaggerModel
from Net import Constants

from utils.dataloader import data_loader
from utils.vocab import Vocab

from settings import config
from utils.opts import train_opts

import logging
import logging.config
logging.config.fileConfig("./settings/logging.ini")


def eval_epoch(model, version, valid_loader, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                valid_loader, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(
                lambda x: x.to(device), batch)
            gold = tgt_seq
            if version in ['V1', 'V4']:
                pred = model(src_seq)
            else:
                # gold = tgt_seq[:, 1:]
                pred = model(src_seq, tgt_seq=tgt_seq)

            # forward
            # pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = model.criterion.cal_performance(pred, gold)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)

            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train_epoch(model, version, train_loader, optimizer, device):
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
        train_loader, mininterval=2, desc='  - (Training)   ', leave=False
    ):
        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        # print(src_seq.size(), tgt_seq.size())
        gold = tgt_seq
        if version in ['V1', 'V4']:
            pred = model(src_seq)
        else:
            pred = model(src_seq, tgt_seq=tgt_seq)

        # forward
        optimizer.zero_grad()
        # pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
        # backward
        loss, n_correct = model.criterion.cal_performance(pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, train_loader, valid_loader, optimizer, opt):

    valid_accus = []
    for epoch in range(opt.epochs):
        start = time.time()
        #  (1) train for one epoch on the training set
        train_loss, train_accu = train_epoch(
            model, opt.version, train_loader, optimizer, opt.device
        )
        logging.info(f'[{epoch}] - (Training  )  ppl: {round(math.exp(min(train_loss, 100)), 2)}, accuracy: {round(100*train_accu, 2)} %, elapse: {int(time.time()-start)} s')

        start = time.time()
        #  (2) evaluate on the validation set
        valid_loss, valid_accu = eval_epoch(model, opt.version, valid_loader, opt.device)
        logging.info(f'[{epoch}] - (Validation)  ppl: {round(math.exp(min(valid_loss, 100)), 2)}, accuracy: {round(100*valid_accu, 2)} %, elapse: {int(time.time()-start)} s')

        valid_accus += [valid_accu]

        checkpoint = {
            "model": model.state_dict()
        }

        if opt.model:
            model_name = opt.model
            if valid_accu >= max(valid_accus):
                torch.save(checkpoint, model_name)
                logging.info(f'Checkpoint {model_name} {valid_accu} updated.')


def main(opt):

    # ========= Loading OPT ========= #
    opt = config.cfg(opt, opt.version)
    # ========= Loading Dataset ========= #
    print(opt)

    logging.info(f"Loading dataset vocab {opt.vocab} ...")
    vocab = Vocab(opt)
    logging.info(f"Loading dataset train {opt.train} ...")
    train_data = torch.load(opt.train)
    logging.info(f"Loading dataset train {opt.valid} ...")
    valid_data = torch.load(opt.valid)

    train_loader = data_loader(vocab, train_data, opt.batch_size, True)
    valid_loader = data_loader(vocab, valid_data, opt.batch_size)

    # ========= Loading Modules ========= #
    model = TaggerModel(opt, vocab).to(opt.device)

    if opt.model and os.path.exists(opt.model):
        logging.info(f"Loading model {opt.model}...")
        checkpoint = torch.load(opt.model)
        model.load_state_dict(checkpoint['model'])
        model.to(opt.device)
    else:
        logging.info(f"{opt.model} Not Exist!")

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           betas=(0.9, 0.98), eps=1e-9)

    train(model, train_loader, valid_loader, optimizer, opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", '--num', type=int, default=100, help="input num")
    parser.add_argument("-v", '--version', required=True, help="version: v1:gru, v2:seq2seq")
    train_opts(parser)
    args = parser.parse_args()
    main(args)
