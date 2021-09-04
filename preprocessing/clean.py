# -*- coding: utf-8 -*-

import re
from typing import List, Tuple

number_regexp = re.compile(r'[0-9０-９]+')
hiragane_regexp = re.compile(r'[ぁ-ん]')
brackets_regexp = re.compile(r'\(.+?\)|{.+?}|\[.+?\]|（.+?）|『.+?』|「.+?」|【.+?】|［.+?］|｢.+?｣|｛.+?｝')


def clean_number(train: Tuple[int, str]) -> Tuple[int, str]:
    label, sentence = train
    return label, number_regexp.sub('0', sentence)


def is_include_hiragana(sentence: str) -> bool:
    return hiragane_regexp.search(sentence) is not None


def validate_clean_sentences(train: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return [(label, sentence) for label, sentence in train if is_include_hiragana(sentence)]


def clean_in_parentheses(train: Tuple[int, str]) -> Tuple[int, str]:
    label, sentence = train
    return label, brackets_regexp.sub('', sentence)
