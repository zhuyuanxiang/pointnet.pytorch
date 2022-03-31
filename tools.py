# -*- encoding: utf-8 -*-
"""
=================================================
@path   : pointnet.pytorch -> tools.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-12-23 15:11
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
import logging
from datetime import datetime

# 默认的warning级别，只输出warning以上的
# 使用basicConfig()来指定日志级别和相关信息
logging.basicConfig(
        level=logging.DEBUG,
        filename='./log/demo.log',
        filemode='w',
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
)


def show_subtitle(message):
    # 输出运行模块的子标题
    print('-' * 15, '>' + message + '<', '-' * 15)
    pass


def show_title(message):
    # 输出运行模块的子标题
    print()
    print('=' * 15, '>' + message + '<', '=' * 15)
    pass


def log_title(message):
    message = '=' * 15 + '>' + message + '<' + '=' * 15
    logging.info(message)


def log_subtitle(message):
    message = '-' * 15 + '>' + message + '<' + '-' * 15
    logging.info(message)


def log_debug(message):
    logging.debug(message)


def log_info(message):
    logging.info(message)


def beep_end():
    # 运行结束的提醒
    import winsound
    winsound.Beep(600, 500)
    pass


def main(name):
    print(f'Hi, {name}', datetime.now())
    show_title("title")
    show_subtitle("subtitle")
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
