#!/bin/bash
# -*- coding:utf-8 -*-
############################################
# File Name    : env.sh
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-08-08
# Last Modified: 2019-08-08 16:09:01
# Descption    :
############################################

# 显示命令
set -x
# 报错即刻退出
set -e

# source activate sl

pip install pipreqs
pipreqs ./ --force
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
