#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : config.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-13
# Last Modified: 2019-08-11 18:57:31
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import toml
import torch

import logging


class Configurable(object):
    def __init__(self, version):
        filename = f"./settings/{version}.toml"
        self.cfg = toml.load(open(filename))

    def to_opt(self, parser):
        for n, net in self.cfg.items():
            setattr(parser, n, net)
        return parser


def cfg(opt, version):
    logging.info(f"Loading opt...")
    opt = Configurable(version).to_opt(opt)
    logging.info(f"Loading opt Done!")

    opt.device = torch.device("cpu")
    if torch.cuda.is_available() and not opt.gpuid:
        print("WARNING: You have a CUDA device, should run with -gpus 0")

    if opt.gpuid:
        torch.cuda.set_device(opt.gpuid[0])
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)
        opt.device = torch.device("cuda")
    return opt


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
