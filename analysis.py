#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : analysis.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-03-07
# Last Modified: 2019-08-12 11:10:20
# Descption    : This script handling the training process.
# Version      : Python 3.7
############################################
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import torch
from models.Translator import Translator
from utils.dataloader import data_loader
from utils.vocab import Vocab
from Net import Constants

from settings import config

import logging
import logging.config
logging.config.fileConfig("./settings/logging.ini")


def test(model, test_loader, vocab, device):
    ''' Epoch operation in evaluation phase '''

    n_word_total = 0
    n_word_correct = 0

    label, y_pred = [], []
    for batch in tqdm(
            test_loader, mininterval=2,
            desc='  - (Test) ', leave=True):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)

        gold = tgt_seq
        # forward
        pred = model.translate(src_seq)

        n_correct = pred.eq(gold)
        non_pad_mask = gold.ne(Constants.PAD)

        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

        # print(n_correct, n_word)
        for idx_preds, idx_tgts in zip(pred.tolist(), gold.tolist()):
            tgtline = [vocab.tgt_idx2word[idx] for idx in idx_tgts]
            pos_index = tgtline.index(Constants.EOS_WORD)
            predline = [vocab.tgt_idx2word[idx] for idx in idx_preds]

            label.extend(tgtline[1: pos_index])
            y_pred.extend(predline[1: pos_index])
    print(f"tgt : {tgtline}\ngold: {predline}")
    accuracy = n_word_correct/n_word_total
    print(f"accuracy={accuracy}")
    labels = ['，', '。', '？']
    print(classification_report(label, y_pred, labels))
    # print(confusion_matrix(label, y_pred))


def main(opt):
    # ========= Loading OPT ========= #
    opt = config.cfg(opt, opt.version)
    # ========= Loading Dataset ========= #
    print(opt)

    logging.info(f"Loading dataset vocab {opt.vocab} ...")
    vocab = Vocab(opt)
    logging.info(f"loading dataset Test {opt.test} ...")
    test_data = torch.load(opt.test)

    test_loader = data_loader(vocab, test_data, opt.batch_size)

    # ========= Loading Modules ========= #
    logging.info(f"Loading model...")
    logging.info(f"src_vocab_size {vocab.src_vocab_size} tgt_vocab_size {vocab.tgt_vocab_size}")
    print(vocab.tgt_idx2word)

    translator = Translator(opt, vocab, opt.device)

    test(translator, test_loader, vocab, opt.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--test', '-test', required=True,
                        help="Test File path from preprocess.py")

    parser.add_argument('--model', '-model', default='model.chkpt',
                        help="Model filename (the model will be saved as "
                        "<save_model>_N.pt where N is the number "
                        "of steps")
    parser.add_argument("-v", '--version', required=True,
                        help="version: v1:gru, v2:seq2seq")
    parser.add_argument('--vocab', '-vocab', default="vocab.pt",
                        help="Path to an existing source vocabulary.")
    parser.add_argument('--gpuid', '-gpuid', default=[], nargs='*', type=int,
                        help="Deprecated see world_size and gpu_ranks.")
    parser.add_argument('--seed', '-seed', type=int, default=3435,
                        help="Random sed")

    parser.add_argument('--batch_size', '-batch_size', type=int, default=64,
                        help='Maximum batch size for training')

    args = parser.parse_args()
    main(args)
