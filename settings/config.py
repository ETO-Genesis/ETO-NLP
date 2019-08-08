#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : config.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-13
# Last Modified: 2019-08-08 13:49:52
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import toml

import logging


class Configurable(object):
    def __init__(self, version):
        filename = f"./settings/{version}.toml"
        self.cfg = toml.load(open(filename))

    def to_opt(self, parser):
        for n, net in self.cfg.items():
            setattr(parser, n, net)
        return parser


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
