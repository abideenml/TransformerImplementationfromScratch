import time
import os
import enum


import torch 
from torchtext.data import Dataset, BucketIterator, Field, Example
from torchtext.data.utils import interleave_keys
from torchtext import datasets
import spacy


BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, DATA_DIR_PATH =''


class DatasetType(enum.Enum):
    IWSLT = 0,
    WMT14 = 1


class LanguageDirection(enum.Enum):
    E2G = 0,
    G2E = 1
