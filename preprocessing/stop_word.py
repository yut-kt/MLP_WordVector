# -*- coding: utf-8 -*-

from glob import glob
from multiprocessing import Pool, cpu_count
from functools import reduce
from operator import or_


def read_stop_words(path: str) -> set:
    """
    :param path: 入力ファイルパス
    :return: stop word set
    """
    with open(path) as f:
        return set([line for line in f.read().splitlines() if line])


def get_stop_words(pathname='./preprocessing/storage/stop_word/*') -> set:
    """
    :param pathname: 入力ファイルパス(ワイルドカード可)
    :return: stop words set
    """
    with Pool(cpu_count()) as p:
        return reduce(or_, p.map(read_stop_words, glob(pathname=pathname)), set())
