# -*- coding: utf-8 -*-

from file_io import read
import preprocessing.scenario

import logging
from datetime import datetime
from argparse import ArgumentParser
from glob import glob
from os.path import join, dirname
import pickle

from gensim.models import KeyedVectors


def preprocess():
    def get_vector(train):
        label, preprocessed_sentence = train
        return label, [model[word] for word in preprocessed_sentence.split() if word in model]

    train_list = read.trains(glob(join(dirname(__file__), 'storage/train/*.list')))
    train_list = preprocessing.scenario.run(train_list)

    model = KeyedVectors.load(wv_model_path)

    with open('storage/train/train.pkl', mode='wb') as f:
        pickle.dump([get_vector(train) for train in train_list], f)


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

    wv_model_path = args.model

    if args.phase == 0:
        preprocess()
    if args.phase == 1:
        train()
