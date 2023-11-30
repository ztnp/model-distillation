# -*- coding:utf-8 -*-
import os
import random

import numpy as np


class Processing(object):

    def __init__(self, test_model=False):
        self.datasets = ['clothing', 'fruit', 'hotel', 'pda', 'shampoo']

        if test_model:
            self.current_path = os.getcwd()
        else:
            self.current_path = os.path.join(os.getcwd(), 'data')

    def get_dataset_list(self):
        return self.datasets

    def __get_datas(self, filepath):
        with open(os.path.join(self.current_path, filepath), 'r', encoding='utf-8') as f:
            content = [
                x
                .strip()
                .replace('\t', '$#Q*&#*&$^')
                .replace('$#Q*&#*&$^', '\t', 1)
                .replace('$#Q*&#*&$^', '') for x in f.readlines()]
        random.shuffle(content)
        return content

    def get_dataset_length_list(self, dataset):
        filepath = os.path.join(dataset, f'{dataset}.txt')
        content = self.__get_datas(filepath)

        length_list = []
        for item in content:
            lines = [x.strip() for x in item.split('\t')]
            length_list.append(len(lines[1]))

        return length_list

    def get_train_examples(self, dataset):
        filepath = os.path.join(dataset, 'train.txt')
        return self.__get_datas(filepath)

    def get_test_example(self, dataset):
        filepath = os.path.join(dataset, 'test.txt')
        return self.__get_datas(filepath)

    def get_dev_example(self, dataset):
        filepath = os.path.join(dataset, 'dev.txt')
        return self.__get_datas(filepath)
