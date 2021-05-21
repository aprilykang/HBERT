import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from Hbert_trainer import BertTrainer as Trainer
from Hbert_dataset import TextClassificationDataset, TextClassificationCollator
from Hbert_utils import read_text
from Hbert_model import Hbert

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', require = True)
    p.add_argument('--train_fn', require = True)
    p.add_argument('--val_fn', require = True)
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-large')
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--hidden', type=int, default=1024)
    p.add_argument('--n_layer', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--warmup_ratio', type=float, default=5e-2)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config


def get_loaders(fn, vfn, tokenizer):
    # Get list of labels and list of texts.
    labels_t, texts_t = read_text(fn)

    # Generate label to index map.
    index_to_label = {
        0: "none",
        1: "offensive",
        2: "hate"
    }

    # valid dataset
    labels_v, texts_v = read_text(vfn)

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        TextClassificationDataset(texts_t, labels_t),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts_v, labels_v),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader, index_to_label


def get_optimizer(model, config):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.lr,
        eps=config.adam_epsilon
    )

    return optimizer


def main(config):
    # Get pretrained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label = get_loaders(
        config.train_fn,
        config.val_fn,
        tokenizer,
        # valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    model = Hbert(
        config.pretrained_model_name,
        len(index_to_label),
        hidden_size=config.hidden,
        n_layers=config.n_layer,
        dropout_p=.25,
    )

    optimizer = get_optimizer(model, config)

    # By default, model returns a hidden representation before softmax func.
    # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
