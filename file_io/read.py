# -*- coding: utf-8 -*-

from typing import List
from multiprocessing import Pool, cpu_count
from itertools import chain


def __extract_sentence(line: str) -> tuple:
    try:
        label, _, sentence = line.split()
        if int(label) < 0:
            label = 0
        return (int(label), sentence)
    except Exception:
        return ()


def __extract_trains(path: str) -> List[tuple]:
    with open(path) as f:
        return [sentence for sentence in map(__extract_sentence, f.read().splitlines()) if sentence]


def trains(paths: List[str]) -> List[tuple]:
    sentences_list = list(map(__extract_trains, paths))

    return list(chain.from_iterable(sentences_list))
