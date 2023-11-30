import os
import random


def main():
    datasets = ['clothing', 'fruit', 'hotel', 'pda', 'shampoo']
    for item in datasets:
        with open(os.path.join(item, '{}.txt'.format(item)), 'r', encoding='utf-8') as f:
            datas = [x.strip() for x in f.readlines()]

        random.shuffle(datas)

        with open(os.path.join(item, 'train.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(datas[:int(0.8 * len(datas))]))

        with open(os.path.join(item, 'dev.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(datas[int(0.8 * len(datas)):int(0.9 * len(datas))]))

        with open(os.path.join(item, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(datas[int(0.9 * len(datas)):]))


if __name__ == '__main__':
    main()
