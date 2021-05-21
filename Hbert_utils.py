import pandas as pd
import re
import emoji
from soynlp.normalizer import repeat_normalize
import torch
import random



def read_text(fn):

    def preprocess_dataframe(df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['comments'] = df['comments'].map(lambda x: clean(str(x)))
        label2index = {
            "none": 0,
            "offensive": 1,
            "hate": 2
        }
        df['label'] = df['label'].replace(label2index)
        return df

    with open(fn, 'r') as f:
        df = pd.read_csv(fn)
        df=preprocess_dataframe(df)
        labels = df.label.to_list()
        texts = df.comments.to_list()

    return labels, texts


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
