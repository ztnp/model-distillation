# -*- coding:utf-8 -*-
import csv
import os
import random
from math import ceil

import jieba
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.Processing import Processing


class Config(object):

    def __init__(self):
        self.lr = 0.001
        self.epochs = 10
        self.batch_size = 32
        self.dropout_num = 0.2

        self.num_filters = 4
        self.filter_sizes = [2, 3, 4]
        self.embedding_size = 128

        self.stopword_filepath = './data/stopword/hit_stopwords.txt'
        self.device = torch.device(
            f'cuda:{random.randint(0, torch.cuda.device_count() - 1)}' if torch.cuda.is_available() else 'cpu')

        self.max_len = 0
        self.vocab_size = 0
        self.output_num = 2

        self.model_filepath = os.path.join(os.getcwd(), 'model', 'model.bin')
        model_base_filepath = os.path.split(self.model_filepath)[0]
        if not os.path.exists(model_base_filepath):
            os.mkdir(model_base_filepath)

        self.model_info_filepath = os.path.join(model_base_filepath, 'evaluate.csv')


class CleanupDataset(object):

    def __init__(self, datas, stopword_filepath):

        self.stopword_filepath = stopword_filepath
        self.stopword = self.load_stopword()

        datas = self.parser_data(datas)
        self.vocabs = self.get_vocabs(datas)

    def get_vocab_size(self):
        return len(self.vocabs)

    def parser_data(self, datas):
        for line in datas:
            lines = [i.strip() for i in line.strip().split('\t')]
            lines.reverse()
            yield lines

    def load_stopword(self):
        with open(self.stopword_filepath, 'r', encoding='utf-8') as f:
            content = [i.strip() for i in f.readlines()]
        return set(content)

    def get_vocabs(self, datas):
        vocabs = set()
        for item in datas:
            for token in jieba.cut(item[0]):
                if token not in self.stopword:
                    vocabs.add(token)
        # vocabs
        vocabs = list(vocabs)
        vocabs.insert(0, '<PAD>')
        vocabs.insert(1, '<UNK>')
        return vocabs

    def convert_ids(self, datas):
        vocab_ids = {vocab: ids for ids, vocab in enumerate(self.vocabs)}

        dataset = []
        for item in datas:
            x = [vocab_ids.get(x, vocab_ids.get('<UNK>')) for x in jieba.cut(item[0]) if x not in self.stopword]
            y = int(item[1])
            dataset.append([x, y])
        return dataset

    def __call__(self, datas, *args, **kwargs):
        datas = self.parser_data(datas)
        return self.convert_ids(datas)


class DatasetGeneration(Dataset):

    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        return self.datas[item]

    def __len__(self):
        return len(self.datas)


class CollateFunction(object):

    def __init__(self, config):
        self.config = config
        self.label_eye = torch.eye(config.output_num)

    def __call__(self, data, *args, **kwargs):
        input_x, input_y = zip(*data)

        train_x = [[0] if not i else i for i in input_x]
        train_x = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in train_x], batch_first=True)
        # train_x = [torch.tensor(i[:self.config.max_len + 1]) for i in input_x]
        # train_x = torch.nn.utils.rnn.pad_sequence(train_x, batch_first=True)

        train_y = self.label_eye[list(input_y)]

        return train_x, train_y


class TextCNN(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)

        self.conv1ds = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_size))
             for k in config.filter_sizes])

        self.dropout = nn.Dropout(config.dropout_num)
        self.fc = nn.Linear(in_features=config.num_filters * len(config.filter_sizes),
                            out_features=config.output_num)

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)

        out = [F.relu(conv(out)).squeeze(3) for conv in self.conv1ds]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)

        out = self.dropout(out)
        out = self.fc(out)

        return out


def load_dataset(stopword_filepath, dataset='hotel', q=90, return_length_list=False):
    # load dataset
    processor = Processing()
    dev_set = processor.get_dev_example(dataset=dataset)
    test_set = processor.get_test_example(dataset=dataset)
    train_set = processor.get_train_examples(dataset=dataset)
    length_list = processor.get_dataset_length_list(dataset=dataset)

    # cleanup dataset
    cleaner = CleanupDataset(
        datas=train_set + dev_set + test_set,
        stopword_filepath=stopword_filepath)

    train_set = cleaner(train_set)
    dev_set = cleaner(dev_set)
    test_set = cleaner(test_set)
    vocab_size = cleaner.get_vocab_size()

    # return dataset
    if return_length_list:
        return train_set, dev_set, test_set, vocab_size, length_list
    return train_set, dev_set, test_set, vocab_size, int(np.percentile(length_list, q))


def evaluate(model, device, collate_fn, data_set):
    acc_value = 0
    rec_value = 0
    f1_value = 0

    interval = 100
    lf = len(data_set)
    iter_num = ceil(lf / interval)
    for i in range(0, lf, interval):
        tmp = data_set[i:i + interval]
        data_x, data_y = collate_fn(tmp)
        data_x = data_x.to(device)
        data_y = torch.argmax(data_y, dim=1).numpy()

        y_pred = model(data_x)
        y_pred = torch.argmax(y_pred, dim=1).to('cpu').detach().numpy()

        acc_value += accuracy_score(data_y, y_pred)
        rec_value += recall_score(data_y, y_pred, average='weighted')
        f1_value += f1_score(data_y, y_pred, average='weighted')

    acc_value /= iter_num
    rec_value /= iter_num
    f1_value /= iter_num
    return acc_value, rec_value, f1_value


def main():
    config = Config()
    train_set, dev_set, test_set, config.vocab_size, config.max_len, = load_dataset(config.stopword_filepath)

    # train set
    collate_fn = CollateFunction(config)
    train_dataloader = DataLoader(
        dataset=DatasetGeneration(train_set),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn)

    # model init
    model = TextCNN(config)
    model.to(config.device)

    # loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # training
    num = 1
    # score = 0
    model.train()
    acc_value, rec_value, f1_value = 0, 0, 0
    for epoch in range(1, config.epochs + 1):
        with tqdm(train_dataloader) as t_epoch:
            for train_x, train_y in t_epoch:
                t_epoch.set_description(f'epoch: {epoch}/{config.epochs}')

                train_x, train_y = train_x.to(config.device), train_y.to(config.device),
                output = model(train_x)

                loss = loss_fn(output, train_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if num % 10 == 0:
                    model.eval()
                    acc_value, rec_value, f1_value = evaluate(model, config.device, collate_fn, dev_set)
                    # tmp_score = acc_value + rec_value + f1_value
                    # if tmp_score > score:
                    #     score = tmp_score
                    #     torch.save(model.state_dict(), config.model_filepath)
                    #     with open(config.model_info_filepath, 'w', newline='', encoding='utf-8') as f:
                    #         writer = csv.writer(f)
                    #         writer.writerow(['acc', 'rec', 'f1'])
                    #         writer.writerow([acc_value, rec_value, f1_value])
                    model.train()

                t_epoch.set_postfix(loss=loss.item(), acc=acc_value, rec=rec_value, f1=f1_value)
                num += 1

    model.eval()
    acc_value, rec_value, f1_value = evaluate(model, config.device, collate_fn, test_set)
    print(f'training finished. test evaluate: acc: {acc_value}, rec: {rec_value}, f1: {f1_value}')

    # torch.save(model.state_dict(), config.model_filepath)
    # print('save model.')


if __name__ == '__main__':
    main()
