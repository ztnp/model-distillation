# -*- coding:utf-8 -*-
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
from transformers import BertModel, BertConfig, BertTokenizer

from data.Processing import Processing


# SEED = 1
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)


class Config(object):

    def __init__(self):
        # distillation alpha
        self.alpha = 0.5

        # BERT
        self.model_name = 'bert-base-chinese'

        # TextCNN
        self.lr = 0.004
        self.epochs = 10
        self.max_len = 64
        self.batch_size = 64
        self.dropout_num = 0.2

        self.num_filters = 4
        self.filter_sizes = [2, 3, 4]
        self.embedding_size = 128
        self.stopword_filepath = './data/stopword/hit_stopwords.txt'

        self.vocab_size = 0
        self.output_num = 2

        # CPU or GPU
        self.device = torch.device(
            f'cuda:{random.randint(0, torch.cuda.device_count() - 1)}' if torch.cuda.is_available() else 'cpu')

        # teacher model bin
        self.teacher_model_filepath = './model/model.bin'


class DLTokenizer(object):

    def __init__(self, datas, stopword_filepath):

        self.stopword_filepath = stopword_filepath
        self.stopword = self.load_stopword()

        datas = self.parser_data(datas)
        self.vocabs = self.get_vocabs(datas)
        self.vocab_ids = {vocab: ids for ids, vocab in enumerate(self.vocabs)}

    def get_vocab_size(self):
        return len(self.vocabs)

    # def get_length_list(self, datas):
    #     return [len(i) for i in datas]
    #

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

    def __call__(self, datas, *args, **kwargs):
        return [[self.vocab_ids.get(j, self.vocab_ids.get('<UNK>'))
                 for j in jieba.cut(i) if j not in self.stopword] for i in datas]


class DatasetGeneration(Dataset):

    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        return self.datas[item]

    def __len__(self):
        return len(self.datas)


class CollateFunction(object):

    def __init__(self, config, dl_tokenizer):
        self.config = config
        self.dl_tokenizer = dl_tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)

    def __call__(self, data, do_bert_tokenizer=True, *args, **kwargs):
        input_y, input_x = zip(*[i.strip().split('\t') for i in data])

        # dl_data_x = self.dl_tokenizer(input_x)
        dl_data_x = [[0] if not i else i for i in self.dl_tokenizer(input_x)]

        dl_data_x = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in dl_data_x], batch_first=True)
        # dl_data_x = [torch.tensor(i[:self.config.max_len + 1]) for i in dl_data_x]
        # dl_data_x = torch.nn.utils.rnn.pad_sequence(dl_data_x, batch_first=True)

        data_y = torch.tensor([int(i) for i in input_y])

        if do_bert_tokenizer:
            bert_data_x = self.tokenizer(
                input_x,
                max_length=self.config.max_len,
                padding=True,
                truncation=True,
                return_tensors='pt')
            return bert_data_x, dl_data_x, data_y
        return dl_data_x, data_y


class BertClassification(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_config = BertConfig.from_pretrained(config.model_name)
        self.bert = BertModel.from_pretrained(config.model_name)

        self.activation = F.relu
        self.fc1 = nn.Linear(model_config.hidden_size, 128)
        self.fc2 = nn.Linear(128, config.output_num)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output_state = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)

        out = self.fc1(output_state[0][:, 0, :])
        out = self.activation(out)
        out = self.fc2(out)
        return out


class TextCNNClassification(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)

        self.conv1ds = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_size))
             for k in config.filter_sizes])

        self.dropout = nn.Dropout(config.dropout_num)
        self.fc = nn.Linear(
            in_features=config.num_filters * len(config.filter_sizes),
            out_features=config.output_num)

    def forward(self, data):
        out = self.embedding(data)
        out = out.unsqueeze(1)

        out = [F.relu(conv(out)).squeeze(3) for conv in self.conv1ds]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, dim=1)

        out = self.dropout(out)
        out = self.fc(out)

        return out


def load_dataset(dataset='hotel'):
    # load dataset
    processor = Processing()
    dev_set = processor.get_dev_example(dataset=dataset)
    test_set = processor.get_test_example(dataset=dataset)
    train_set = processor.get_train_examples(dataset=dataset)
    length_list = processor.get_dataset_length_list(dataset=dataset)

    return train_set, dev_set, test_set, length_list


def evaluate(model, device, collate_fn, data_set):
    acc_value = 0
    rec_value = 0
    f1_value = 0

    interval = 100
    lf = len(data_set)
    iter_num = ceil(lf / interval)
    for i in range(0, lf, interval):
        tmp = data_set[i:i + interval]
        data_x, data_y = collate_fn(tmp, do_bert_tokenizer=False)
        data_x = data_x.to(device)
        # data_y = torch.argmax(data_y, dim=1).numpy()

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
    train_set, dev_set, test_set, length_list = load_dataset()

    # init tokenizer
    dl_tokenizer = DLTokenizer(
        datas=train_set + dev_set + test_set,
        stopword_filepath=config.stopword_filepath)
    config.vocab_size = dl_tokenizer.get_vocab_size()

    # dataloader z
    collate_fn = CollateFunction(config, dl_tokenizer)
    train_dataloader = DataLoader(
        dataset=DatasetGeneration(train_set),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn)

    # init model
    s_model = TextCNNClassification(config).to(config.device)

    t_model = BertClassification(config)
    t_model.load_state_dict(torch.load(config.teacher_model_filepath))
    # t_model.load_state_dict(torch.load(config.teacher_model_filepath, map_location='cpu'))
    t_model.to(config.device)

    # optimizer and loss_fn
    ce_loss = nn.NLLLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(s_model.parameters(), lr=config.lr)

    num = 0
    for epoch in range(1, config.epochs + 1):
        with tqdm(train_dataloader) as t_epoch:
            for bert_data_x, dl_data_x, data_y in t_epoch:
                t_epoch.set_description(f'epoch: {epoch}/{config.epochs}')

                bert_data_x, dl_data_x, data_y = bert_data_x.to(config.device), dl_data_x.to(config.device), data_y.to(
                    config.device)

                with torch.no_grad():
                    t_output = t_model(
                        input_ids=bert_data_x['input_ids'],
                        token_type_ids=bert_data_x['token_type_ids'],
                        attention_mask=bert_data_x['attention_mask'])
                    t_output = F.softmax(t_output, dim=1)

                s_output = s_model(dl_data_x)

                loss = config.alpha * ce_loss(F.log_softmax(s_output, dim=1), data_y) + \
                       (1 - config.alpha) * mse_loss(t_output, F.softmax(s_output, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if num % 10 == 0:
                    s_model.eval()
                    acc_value, rec_value, f1_value = evaluate(s_model, config.device, collate_fn, dev_set)
                    s_model.train()

                t_epoch.set_postfix(loss=loss.item(), acc=acc_value, rec=rec_value, f1=f1_value)
                num += 1

    s_model.eval()
    acc_value, rec_value, f1_value = evaluate(s_model, config.device, collate_fn, test_set)
    print(f'training finished. test evaluate: acc: {acc_value}, rec: {rec_value}, f1: {f1_value}')


if __name__ == '__main__':
    main()
