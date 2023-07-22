# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/16 8:39
@Auth ： zy

"""
import random
import numpy as np
# list = ['AGCTT', 'ACCGT', 'AACGT']
# lable = np.array([[1],[0],[1]])
#
# Q = np.array(list)
import xlrd
def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)

    booksheet_pos = workbook.sheet_by_index(0)
    nrows_pos = booksheet_pos.nrows

    booksheet_neg = workbook.sheet_by_index(1)
    nrows_neg = booksheet_neg.nrows

    seq = []
    label = []
    for i in range(nrows_pos):
        seq.append(booksheet_pos.row_values(i)[0])
        label.append(booksheet_pos.row_values(i)[1])
    for j in range(nrows_neg):
        seq.append((booksheet_neg.row_values(j)[0]))
        label.append(booksheet_neg.row_values(j)[1])

    return seq, np.array(label).astype(int)

def ACGTto0123(filename):
    seq, label = read_seq_label(filename)
    seq0123 = []
    for i in range(len(seq)):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq0123.append(one_seq)
    return seq0123, label

def seq_to01_to0123(filename):

    seq, label = read_seq_label(filename)

    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq_start = 0
        for j in range(seq_len):
            seq_0123[i, j] = int(one_seq[j - seq_start])
            if j < seq_start:
                seq_01[i, j, :] = 0.25
            else:
                try:
                    seq_01[i, j, int(one_seq[j - seq_start])] = 1
                except:
                    seq_01[i, j, :] = 0.25
    return seq_01, seq_0123, label

def load_data(filename):

    seq01, seq_0123, label = seq_to01_to0123(filename)

    r = random.random
    random.seed(2)
    a = np.linspace(0, len(label)-1, len(label)).astype(int)
    random.shuffle(a, random=r)

    num_total = len(label)
    num_train = int(len(label)*0.8)
    num_val = int(len(label)*0.1)
    num_test = num_total - num_train - num_val

    train_index = a[:num_train]
    valid_index = a[num_train:num_train+num_val]
    test_index = a[num_train+num_val:num_total]

    x_train = seq01[train_index, :, :]
    x_val = seq01[valid_index, :, :]
    x_test = seq01[test_index, :, :]

    y_train = label[train_index]
    y_val = label[valid_index]
    y_test = label[test_index]

    # # 这一段保存测试数据
    # seq, label = read_seq_label(filename)
    # change = np.array(seq)
    # x_test_save = change[test_index].tolist()
    # y_test_save = label[test_index]
    # from pandas import DataFrame
    # df = DataFrame({'x_test_save': x_test_save, 'y_test_save': y_test_save},
    #                index=range(num_test))
    # df.to_csv('data/testdata_CSV_write.csv', index=False)









    return x_train, y_train, x_val, y_val, x_test, y_test



def read_test_txt(data_path):

    seq_name = []
    seq = []

    with open(data_path, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            if line.startswith('>'):
                line = line[0:-1]
                seq_name.append(line)
            elif line.strip() == "":
                pass
            else:
                line = line[0:41]
                seq.append(line)
    return seq_name, seq

def read_test_txt_to01_to0123(data_path):

    seq_name, seq = read_test_txt(data_path)

    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq_start = 0
        for j in range(seq_len):
            seq_0123[i, j] = int(one_seq[j - seq_start])
            if j < seq_start:
                seq_01[i, j, :] = 0.25
            else:
                try:
                    seq_01[i, j, int(one_seq[j - seq_start])] = 1
                except:
                    seq_01[i, j, :] = 0.25
    return seq_name, seq , seq_01




if __name__ == '__main__':
    # filename = 'data/A.thaliana.xlsx'
    # load_data(filename)
    data_path = 'D://zyD//00BS\zcodem3_web_root\PredictDataOFUsers//jobid//996//996.txt'
    read_test_txt_to01_to0123(data_path)




