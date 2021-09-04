# -*- coding: utf-8 -*-

import subprocess
import MeCab
from typing import List, Tuple
from multiprocessing import Pool, cpu_count


def get_neologd():
    dicdir = subprocess.run(
        ['mecab-config', '--dicdir'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    ).stdout.decode().strip()

    return f'{dicdir}/mecab-ipadic-neologd'


mecab = MeCab.Tagger(f'-d {get_neologd()}')


def get_wakati(train: Tuple[int, str]) -> Tuple[int, str]:
    label, sentence = train
    wakati = []
    for result in mecab.parse(sentence).split('\n'):
        if result == 'EOS':
            break
        morpheme, morpheme_info = result.split('\t', maxsplit=1)
        word_class, _ = morpheme_info.split(',', maxsplit=1)
        if word_class != '記号':
            wakati.append(morpheme)
    return label, ' '.join(wakati)


def get_wakatis(train: List[Tuple[int, str]]):
    with Pool(cpu_count()) as p:
        return p.map(get_wakati, train)
