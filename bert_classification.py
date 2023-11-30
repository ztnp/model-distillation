# -*- coding:utf-8 -*-
import csv
import os
import random
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig

from data.Processing import Processing


class Config(object):

    def __init__(self):
        self.lr = 0.00001
        self.epochs = 5
        self.max_len = 16
        self.batch_size = 64
        self.output_num = 2
        self.model_name = 'bert-base-chinese'
        self.device = torch.device(
            f'cuda:{random.randint(0, torch.cuda.device_count() - 1)}' if torch.cuda.is_available() else 'cpu')

        self.model_filepath = os.path.join(os.getcwd(), 'model', 'model.bin')
        model_base_filepath = os.path.split(self.model_filepath)[0]
        if not os.path.exists(model_base_filepath):
            os.mkdir(model_base_filepath)

        self.model_info_filepath = os.path.join(model_base_filepath, 'evaluate.csv')


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
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        self.label_eye = torch.eye(config.output_num)

    def __call__(self, data, *args, **kwargs):
        input_x, input_y = zip(*[[x.strip(), int(y.strip())] for y, x in [i.split('\t') for i in data]])

        data_x = self.tokenizer(
            input_x,
            max_length=self.config.max_len,
            padding=True,
            truncation=True,
            return_tensors='pt')

        data_y = self.label_eye[list(input_y)]
        return data_x, data_y


class BertClassificationModel(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = BertConfig.from_pretrained(config.model_name)

        self.bert = BertModel.from_pretrained(config.model_name)
        self.fc1 = nn.Linear(model_config.hidden_size, 128)
        self.activation = F.relu
        self.fc2 = nn.Linear(128, config.output_num)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output_state = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        out = self.fc1(output_state[0][:, 0, :])
        out = self.activation(out)
        out = self.fc2(out)
        return out


def get_dataset(dataset='hotel', q=90, return_length_list=False):
    processor = Processing()
    dev_set = processor.get_dev_example(dataset=dataset)
    test_set = processor.get_test_example(dataset=dataset)
    train_set = processor.get_train_examples(dataset=dataset)
    length_list = processor.get_dataset_length_list(dataset=dataset)

    if return_length_list:
        return train_set, dev_set, test_set, length_list
    return train_set, dev_set, test_set, int(np.percentile(length_list, q))


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

        y_pred = model(
            input_ids=data_x['input_ids'],
            token_type_ids=data_x['token_type_ids'],
            attention_mask=data_x['attention_mask'])
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
    train_set, dev_set, test_set, config.max_len = get_dataset()
    # config.max_len = max_len

    # train set
    train_dataset = DatasetGeneration(train_set)
    collate_fn = CollateFunction(config)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn)

    # model init
    model = BertClassificationModel(config)
    model.to(config.device)

    # loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), config.lr)

    # training
    num = 1
    score = 0
    model.train()
    acc_value, rec_value, f1_value = 0, 0, 0
    for epoch in range(1, config.epochs + 1):
        with tqdm(train_dataloader) as t_epoch:
            for train_x, train_y in t_epoch:
                t_epoch.set_description(f'epoch: {epoch}/{config.epochs}')

                train_x, train_y = train_x.to(config.device), train_y.to(config.device)
                output = model(
                    input_ids=train_x['input_ids'],
                    token_type_ids=train_x['token_type_ids'],
                    attention_mask=train_x['attention_mask'])

                loss = loss_fn(output, train_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if num % 10 == 0:
                    model.eval()
                    acc_value, rec_value, f1_value = evaluate(model, config.device, collate_fn, dev_set)
                    tmp_score = acc_value + rec_value + f1_value
                    if tmp_score > score:
                        score = tmp_score
                        torch.save(model.state_dict(), config.model_filepath)
                        with open(config.model_info_filepath, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['acc', 'rec', 'f1'])
                            writer.writerow([acc_value, rec_value, f1_value])
                    model.train()

                t_epoch.set_postfix(loss=loss.item(), acc=acc_value, rec=rec_value, f1=f1_value)
                num += 1

    model.eval()
    acc_value, rec_value, f1_value = evaluate(model, config.device, collate_fn, test_set)
    print(f'training finished. test evaluate: acc: {acc_value}, rec: {rec_value}, f1: {f1_value}')


if __name__ == '__main__':
    main()
