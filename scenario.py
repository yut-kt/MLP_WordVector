# -*- coding: utf-8 -*-

from file_io import read
import preprocessing.scenario

import logging
from datetime import datetime
from argparse import ArgumentParser
from glob import glob
from os.path import join, dirname
from multiprocessing import cpu_count, Pool

from gensim.models import KeyedVectors
import numpy as np


def get_vector(train):
    label, preprocessed_sentence = train
    return label, [model[word] for word in preprocessed_sentence.split() if word in model]


def padding(train):
    label, features = train
    return label, np.pad(features, ([0, max_len - len(features)], [0, 0]), 'constant')


def preprocess():
    train_list = read.trains(glob(join(dirname(__file__), 'storage/train/*.list')))
    train_list = preprocessing.scenario.run(train_list)

    with Pool(cpu_count()) as p:
        train_list = p.map(get_vector, train_list)

    train_list = [(label, features) for label, features in train_list if features]

    global max_len
    max_len = max(map(lambda train: len(train[1]), train_list))

    with Pool(cpu_count()) as p:
        train_list = p.map(padding, train_list)

    print('ok')

    np.savez('train.npz', train=train_list)


def train():
    # TODO
    pass


if __name__ == '__main__':
    log_formatter = '%(levelname)s : %(asctime)s : %(message)s'
    log_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f'./storage/log/{log_filename}.log', level=logging.DEBUG, format=log_formatter)

    parser = ArgumentParser()
    parser.add_argument('-o', '--model_name', help='モデルの保存名', default='output')
    parser.add_argument('-p', '--phase', help='select mode', type=int, choices=[0, 1], required=True)
    parser.add_argument('-m', '--model', help='word2vec model path', type=str, required=True)
    args = parser.parse_args()

    model = KeyedVectors.load(args.model)

    if args.phase == 0:
        preprocess()
    if args.phase == 1:
        train()
