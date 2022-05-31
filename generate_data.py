import argparse
from datasets import load_dataset
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import string
import re
import unidecode
import collections
import torch
import hashlib
from typing import List, Tuple, Dict, Callable
import os
import json
import numpy as np
import logging
from datasets import load_metric
import spacy
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import csv


class LinearizeWebnlgInput():

    def __init__(self, spacy_pipeline, lowercase=False, format: str ='gem',):
        self.lowercase = lowercase
        self.format = format
        self.spacy_pipeline = spacy_pipeline

    def __call__(self,input: List[str])-> str:

        if self.format != 'gem':
            raise ValueError(f'Unsupported format for now: {self.format}')

        triples = [Triple(triple,
                          spacy_pipeline=self.spacy_pipeline,
                          lower=self.lowercase)
                   for triple in input]

        table = dict()
        for triple in triples:
            table.setdefault(triple.sbj, list())
            table[triple.sbj].append((triple.obj, triple.prp))

        ret = list()
        for entidx, (entname, entlist) in enumerate(table.items(), 1):
            ret.append(f'entity [ {entname} ]')
            for values, key in entlist:
                ret.append(f'{key} [ {values} ]')

        return ' , '.join(ret)

class Triple:
    def __init__(self, raw_text: str, spacy_pipeline, lower: bool = False,):
        sbj, prp, obj = self.safe_split(raw_text)
        obj = ' '.join([t.text for t in spacy_pipeline(self.clean_obj(obj.strip(), lc=lower))])
        prp = self.clean_prp(prp.strip())
        sbj = ' '.join([t.text for t in spacy_pipeline(self.clean_obj(sbj.strip(), lc=lower))])
        if prp == 'ethnicgroup':
            obj = obj.split('_in_')[0]
            obj = obj.split('_of_')[0]

        self.sbj = sbj
        self.obj = obj
        self.prp = prp

    @staticmethod
    def safe_split(raw_text) -> List[str]:

        if not isinstance(raw_text, str):
            raise TypeError('A triple must be a string with two "|"'
                            f'but you gave: {raw_text}')

        split = raw_text.strip().split('|')
        if not len(split) == 3:
            raise TypeError('A triple must be a string with two "|"'
                            f'but you gave: {raw_text}')

        return split

    def __repr__(self):
        return f'{self.sbj} | {self.prp} | {self.obj}'

    @staticmethod
    def clean_obj(s, lc: bool = False):
        s = unidecode.unidecode(s)
        if lc: s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub('_', ' ', s)  # turn undescores to spaces
        return s

    @staticmethod
    def clean_prp(s: str, lc: bool=False) -> str:
        s = unidecode.unidecode(s)
        if lc: s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub('\s+', '_', s)  # turn spaces to underscores
        s = re.sub('\s+\(in metres\)', '_m', s)
        s = re.sub('\s+\(in feet\)', '_f', s)
        s = re.sub('\(.*\)', '', s)
        return s.strip()


train_dataset = load_dataset("web_nlg", 'release_v3.0_en', split='train')
dev_dataset = load_dataset("web_nlg", 'release_v3.0_en', split='dev')
test_dataset = load_dataset("web_nlg", 'release_v3.0_en', split='test')

piped = LinearizeWebnlgInput(spacy_pipeline=spacy.load('en_core_web_sm'))

train_csv = open('train_web_nlg.csv', 'w', encoding="utf-8", newline='')
dev_csv = open('dev_web_nlg.csv', 'w', encoding="utf-8", newline='')
test_csv = open('test_web_nlg.csv', 'w', encoding="utf-8", newline='')

writer = csv.writer(train_csv)
for data in train_dataset:
    for triple_set in data['modified_triple_sets']['mtriple_set']:
        context = piped(triple_set)
        for answer in data['lex']['text']:
            writer.writerow([answer, context])
train_csv.close()
print("train")

writer = csv.writer(dev_csv)
for data in dev_dataset:
    for triple_set in data['modified_triple_sets']['mtriple_set']:
        context = piped(triple_set)
        for answer in data['lex']['text']:
            writer.writerow([answer, context])
dev_csv.close()
print("dev")

writer = csv.writer(test_csv)
for data in test_dataset:
    for triple_set in data['modified_triple_sets']['mtriple_set']:
        context = piped(triple_set)
        for answer in data['lex']['text']:
            writer.writerow([answer, context])
test_csv.close()


