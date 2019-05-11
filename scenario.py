# -*- coding: utf-8 -*-

from file_io import read
import preprocessing.scenario

import logging
from datetime import datetime
from argparse import ArgumentParser
from glob import glob
from os.path import join, dirname
from multiprocessing import cpu_count, Pool
import pickle

from gensim.models import KeyedVectors
import numpy as np
import GPyOpt


def get_vector(train):
    label, preprocessed_sentence = train
    return label, [model[word] for word in preprocessed_sentence.split() if word in model]


def padding(train):
    label, features = train
    features = features[:vector_len]
    return label, np.pad(features, ([0, vector_len - len(features)], [0, 0]), 'constant')


def preprocess():
    train_list = read.trains(glob(join(dirname(__file__), 'storage/train/*.list')))
    train_list = preprocessing.scenario.run(train_list)

    with Pool(cpu_count()) as p:
        train_list = p.map(get_vector, train_list)

    train_list = [(label, features) for label, features in train_list if features]

    with Pool(cpu_count()) as p:
        train_list = p.map(padding, train_list)

    np.savez('train.npz', train=train_list)


def train():
    def run(l1_out=512, l2_out=512, l3_out=512,
            l1_drop=0.2, l2_drop=0.2, l3_drop=0.3,
            bn1=0, bn2=0, bn3=0,
            batch_size=100, epochs=10, validation_split=0.1):
        import mlp
        _mlp = mlp.MLP(l1_out=l1_out, l2_out=l2_out, l3_out=l3_out,
                       l1_drop=l1_drop, l2_drop=l2_drop, l3_drop=l3_drop,
                       bn1=bn1, bn2=bn2, bn3=bn3,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=validation_split)
        mnist_evaluation = _mlp.evaluate()
        return mnist_evaluation

    def f(x):
        print(x)
        evaluation = run(
            l1_out=int(x[:, 0]),
            l2_out=int(x[:, 1]),
            l3_out=int(x[:, 2]),
            l1_drop=float(x[:, 3]),
            l2_drop=float(x[:, 4]),
            l3_drop=float(x[:, 5]),
            bn1=int(x[:, 6]),
            bn2=int(x[:, 7]),
            bn3=int(x[:, 8]),
            batch_size=int(x[:, 9]),
            epochs=int(x[:, 10]),
            validation_split=float(x[:, 11]))
        print("loss:{0} \t\t accuracy:{1}".format(evaluation[0], evaluation[1]))
        return evaluation[0]

    bounds = [
        {'name': 'l1_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024)},
        {'name': 'l2_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024)},
        {'name': 'l3_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024)},
        {'name': 'l1_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
        {'name': 'l2_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
        {'name': 'l3_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
        {'name': 'bn1', 'type': 'discrete', 'domain': (0, 1)},
        {'name': 'bn2', 'type': 'discrete', 'domain': (0, 1)},
        {'name': 'bn3', 'type': 'discrete', 'domain': (0, 1)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (10, 100, 500)},
        {'name': 'epochs', 'type': 'discrete', 'domain': (5, 10, 20)},
        {'name': 'validation_split', 'type': 'continuous', 'domain': (0.0, 0.3)},
    ]

    opt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
    opt.run_optimization(max_iter=15)

    optbounds = {
        'l1_out': opt.x_opt[0],
        'l2_out': opt.x_opt[1],
        'l3_out': opt.x_opt[2],
        'l1_drop': opt.x_opt[3],
        'l2_drop': opt.x_opt[4],
        'l3_drop': opt.x_opt[5],
        'bn1': opt.x_opt[6],
        'bn2': opt.x_opt[7],
        'bn3': opt.x_opt[8],
        'batch_size': opt.x_opt[9],
        'epochs': opt.x_opt[10],
        'validation_split': opt.x_opt[11],
    }

    for key, value in optbounds.items():
        print(key, '->', value)


if __name__ == '__main__':
    log_formatter = '%(levelname)s : %(asctime)s : %(message)s'
    log_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f'./storage/log/{log_filename}.log', level=logging.DEBUG, format=log_formatter)

    parser = ArgumentParser()
    parser.add_argument('-o', '--model_name', help='モデルの保存名', default='output')
    parser.add_argument('-p', '--phase', help='select mode', type=int, choices=[0, 1], required=True)
    parser.add_argument('-m', '--model', help='word2vec model path', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    args = parser.parse_args()

    vector_len = 50

    if args.phase == 0:
        model = KeyedVectors.load(args.model)
        preprocess()
    if args.phase == 1:
        with open(args.dataset, 'rb') as f:
            dataset = pickle.load(f)
        train()
