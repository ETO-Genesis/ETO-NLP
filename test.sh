#!/bin/bash
# -*- coding:utf-8 -*-
############################################
# File Name    : train.sh
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-13
# Last Modified: 2019-07-13 00:17:58
# Descption    :
############################################

# 显示命令
set -x
# 报错即刻退出
set -e

# $1 : language
# $2 : gpuid
# $3 : version

rm models
ln -s $3 models

INST_DIR=./data/$1/insts

MODEL_DIR=./data/$1/model

python analysis.py --vocab ./data/$1/vocab --test ${INST_DIR}/test.pt -gpuid $2 --model ${MODEL_DIR}/$3.chkpt -v $3
