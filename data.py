import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
import matplotlib.pylab as pl
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pdb

def load_mnist_data(dir_path, n_sample_source = 1000, n_sample_targets = 50, n_sample_test = 200, time_length = 5):

    Xs = np.zeros((n_sample_source, 28*28))
    ys = np.zeros((n_sample_source, 1))
    Xt = []
    yt = []

    Xtest = []
    ytest = []

    with open(os.path.join(dir_path, 'train-labels.csv'),'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    img = 0
    for i in range(0, n_sample_source):
        index = i * (time_length + 1)
        img_path = dir_path + rows[index][0]
        img_label = int(rows[index][1])
        img = cv2.imread(img_path, 0).reshape(1, -1)
        Xs[i, :] = img
        ys[i, 0] = img_label

    Xs = Xs / 255
    ys = ys.squeeze()

    for j in range(time_length):
        x = np.zeros((n_sample_targets, 28*28))
        y = np.zeros((n_sample_targets, 1))
        
        for i in range(n_sample_targets):
            index = i * (time_length + 1) + j + 1
            img_path = dir_path + rows[index][0]
            img_label = int(rows[index][1])
            img = cv2.imread(img_path, 0).reshape(1, -1)
            x[i, :] = img
            y[i, 0] = img_label
        
        x = x / 255
        y = y.squeeze()
        Xt.append(x)
        yt.append(y)
    
    for j in range(time_length):
        x = np.zeros((n_sample_test, 28*28))
        y = np.zeros((n_sample_test, 1))
        
        for i in range(n_sample_test):

            index = (i + n_sample_targets) * (time_length + 1) + j + 1
            img_path = dir_path + rows[index][0]
            img_label = int(rows[index][1])
            img = cv2.imread(img_path, 0).reshape(1, -1)
            x[i, :] = img
            y[i, 0] = img_label
        
        x = x / 255
        y = y.squeeze()
        Xtest.append(x)
        ytest.append(y)
    
    return Xs, ys, Xt, yt, Xtest, ytest
    

def load_battery_data(n_samples_source = 67, n_samples_targets = 10, time_length = 4, shuffle_or_not = False):
    dir = '/home/hanbingliu/out_SOC_005-075_excel/'
    s_val = pd.read_excel(dir + 'out-SOC-005.xlsx', engine='openpyxl').values
    s_cnt = n_samples_source if n_samples_source <= s_val.shape[0] else s_val.shape[0]
    Xs = s_val[: s_cnt, 2 : 23].astype(np.float64)
    ys = s_val[: s_cnt, 1].astype(np.float64)

    Xt = []
    yt = []
    Xt_all = []
    yt_all = []
    for i in range(time_length):
        t_dir = dir + 'out-SOC-0{}.xlsx'.format(5*(i+2))
        # if i == 8:
        #     t_dir = dir + 'out-SOC-005.xlsx'
        # else:
        #     t_dir = dir + 'out-SOC-0{}.xlsx'.format(5*(9-i))

        t_val = pd.read_excel(t_dir, engine='openpyxl').values
        X = t_val[:, 2 : 23].astype(np.float64)
        y = t_val[:, 1].astype(np.float64)

        Xt_all.append(X)
        yt_all.append(y)

        rand = np.arange(X.shape[0])
        if shuffle_or_not:
            np.random.shuffle(rand)
                
        t_cnt = n_samples_targets if n_samples_targets <= t_val.shape[0] else t_val.shape[0]
        X1 = X[rand[: t_cnt]]
        y1 = y[rand[: t_cnt]]

        Xt.append(X1)
        yt.append(y1)
    
    return Xs, ys, Xt, yt, Xt_all, yt_all

def load_battery_data_random(n_samples_source = 67, n_samples_targets = 10, time_series = [], shuffle_or_not = True, random_seed = 0):
    dir = '/home/hanbingliu/out_SOC_005-075_excel/'
    s_val = pd.read_excel(dir + 'out-SOC-005.xlsx', engine='openpyxl').values
    s_cnt = n_samples_source if n_samples_source <= s_val.shape[0] else s_val.shape[0]
    Xs = s_val[: s_cnt, 2 : 23].astype(np.float64)
    ys = s_val[: s_cnt, 1].astype(np.float64)

    Xt = []
    yt = []
    Xt_all = []
    yt_all = []
    for i in time_series:
        t_dir = dir + 'out-SOC-0{}.xlsx'.format(i)
        t_val = pd.read_excel(t_dir, engine='openpyxl').values
        X = t_val[:, 2 : 23].astype(np.float64)
        y = t_val[:, 1].astype(np.float64)

        Xt_all.append(X)
        yt_all.append(y)

        rand = np.arange(X.shape[0])
        if shuffle_or_not:
            np.random.seed(random_seed * i)
            np.random.shuffle(rand)
            # print(rand)
        
        t_cnt = n_samples_targets if n_samples_targets <= t_val.shape[0] else t_val.shape[0]
        X1 = X[rand[: t_cnt]]
        y1 = y[rand[: t_cnt]]

        Xt.append(X1)
        yt.append(y1)
    
    return Xs, ys, Xt, yt, Xt_all, yt_all

def load_battery_data_split(n_samples_source = 67, n_samples_targets = 10, time_series = [], shuffle_or_not = True, random_seed = 1, train_set = 20):
    dir = '/home/hanbingliu/out_SOC_005-075_excel/'
    s_val = pd.read_excel(dir + 'out-SOC-005.xlsx', engine='openpyxl').values
    s_cnt = n_samples_source if n_samples_source <= s_val.shape[0] else s_val.shape[0]
    Xs = s_val[: s_cnt, 2 : 23].astype(np.float64)
    ys = s_val[: s_cnt, 1].astype(np.float64)

    s_val = pd.read_excel(dir + 'out-SOC-010.xlsx', engine='openpyxl').values
    x1 = s_val[:train_set, 2 : 23].astype(np.float64)
    y1 = np.ones((train_set, )) * 10
    xt1 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt1_value = s_val[train_set:, 1].astype(np.float64)
    yt1 = np.ones((xt1.shape[0], )) * 10

    s_val = pd.read_excel(dir + 'out-SOC-015.xlsx', engine='openpyxl').values
    x2 = s_val[:train_set, 2 : 23].astype(np.float64)
    y2 = np.ones((train_set, )) * 15
    xt2 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt2_value = s_val[train_set:, 1].astype(np.float64)
    yt2 = np.ones((xt2.shape[0], )) * 15

    s_val = pd.read_excel(dir + 'out-SOC-020.xlsx', engine='openpyxl').values
    x3 = s_val[:train_set, 2 : 23].astype(np.float64)
    y3 = np.ones((train_set, )) * 20
    xt3 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt3_value = s_val[train_set:, 1].astype(np.float64)
    yt3 = np.ones((xt3.shape[0], )) * 20

    s_val = pd.read_excel(dir + 'out-SOC-025.xlsx', engine='openpyxl').values
    x4 = s_val[:train_set, 2 : 23].astype(np.float64)
    y4 = np.ones((train_set, )) * 25
    xt4 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt4_value = s_val[train_set:, 1].astype(np.float64)
    yt4 = np.ones((xt4.shape[0], )) * 25

    s_val = pd.read_excel(dir + 'out-SOC-030.xlsx', engine='openpyxl').values
    x5 = s_val[:train_set, 2 : 23].astype(np.float64)
    y5 = np.ones((train_set, )) * 30
    xt5 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt5_value = s_val[train_set:, 1].astype(np.float64)
    yt5 = np.ones((xt5.shape[0], )) * 30

    s_val = pd.read_excel(dir + 'out-SOC-035.xlsx', engine='openpyxl').values
    x6 = s_val[:train_set, 2 : 23].astype(np.float64)
    y6 = np.ones((train_set, )) * 35
    xt6 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt6_value = s_val[train_set:, 1].astype(np.float64)
    yt6 = np.ones((xt6.shape[0], )) * 35

    s_val = pd.read_excel(dir + 'out-SOC-040.xlsx', engine='openpyxl').values
    x7 = s_val[:train_set, 2 : 23].astype(np.float64)
    y7 = np.ones((train_set, )) * 40
    xt7 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt7_value = s_val[train_set:, 1].astype(np.float64)
    yt7 = np.ones((xt3.shape[0], )) * 40

    s_val = pd.read_excel(dir + 'out-SOC-045.xlsx', engine='openpyxl').values
    x8 = s_val[:train_set, 2 : 23].astype(np.float64)
    y8 = np.ones((train_set, )) * 45
    xt8 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt8_value = s_val[train_set:, 1].astype(np.float64)
    yt8 = np.ones((xt8.shape[0], )) * 45

    s_val = pd.read_excel(dir + 'out-SOC-050.xlsx', engine='openpyxl').values
    x9 = s_val[:train_set, 2 : 23].astype(np.float64)
    y9 = np.ones((train_set, )) * 50
    xt9 = s_val[train_set:, 2 : 23].astype(np.float64)
    yt9_value = s_val[train_set:, 1].astype(np.float64)
    yt9 = np.ones((xt9.shape[0], )) * 50

    x = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9))
    y = np.concatenate((y1, y2, y3, y4, y5, y6, y7, y8, y9))

    xt0 = np.concatenate((xt1, xt2, xt3, xt4, xt5, xt6, xt7, xt8, xt9))
    yt0 = np.concatenate((yt1, yt2, yt3, yt4, yt5, yt6, yt7, yt8, yt9))
    yt_value = np.concatenate((yt1_value, yt2_value, yt3_value, yt4_value, yt5_value, yt6_value, yt7_value, yt8_value, yt9_value))

    knn = KNeighborsClassifier(n_neighbors=2)  
    knn.fit(x, y)
    y_pred = knn.predict(xt0)
    acc = accuracy_score(yt0, y_pred)
    # print("soc prediction score: ", acc)

    Xt_clf = []
    yt_clf = []
    Xt_all = []
    yt_all = []

    Xt_ = []
    yt_ = []

    Xt_random = []
    yt_random = []

    for i in time_series:

        Xt_true = xt0[yt0 == i]
        yt_true = yt_value[yt0 == i]
        Xt_all.append(Xt_true)
        yt_all.append(yt_true)

        Xt_predict = xt0[y_pred == i]
        yt_predict = yt_value[y_pred == i]
        rand = np.arange(Xt_predict.shape[0])
        if shuffle_or_not:
            np.random.seed(random_seed * i + 1)
            np.random.shuffle(rand)
            # print(rand)
        
        t_cnt = n_samples_targets
        X1 = Xt_predict[rand[: t_cnt]]
        y1 = yt_predict[rand[: t_cnt]]

        rand = np.arange(Xt_true.shape[0])
        if shuffle_or_not:
            np.random.seed(random_seed * i)
            np.random.shuffle(rand)

        X2 = Xt_true[rand[: t_cnt]]
        y2 = yt_true[rand[: t_cnt]]

        rand = np.arange(xt0.shape[0])
        if shuffle_or_not:
            np.random.seed(random_seed * i)
            np.random.shuffle(rand)
        
        X3 = xt0[rand[: t_cnt]]
        y3 = yt_value[rand[: t_cnt]]

        if i == time_series[-1]:
            X3 = X2
            y3 = y2
            X1 = X2
            y1 = y2

        Xt_clf.append(X1)
        yt_clf.append(y1)

        Xt_.append(X2)
        yt_.append(y2)
        
        Xt_random.append(X3)
        yt_random.append(y3)

    return Xs, ys, Xt_clf, yt_clf, Xt_all, yt_all, acc, Xt_, yt_, Xt_random, yt_random

    

def load_seq_two_moon_data(n_samples_source, n_samples_targets, time_length, noise=.1, max_angle=90):
    Xs, ys = make_moons(n_samples_source, shuffle=True, noise=noise)

    # Initialize target data
    Xt = []
    yt = []
    angles = []

    for k in range(time_length):
        angle = (np.deg2rad(max_angle) / time_length) * (k + 1)
        X, y = make_moons(n_samples_targets, shuffle=True, noise=noise)
        X = rotate_2d(X, angle)

        Xt.append(X)
        yt.append(y)
        angles.append(np.rad2deg(angle))

    return Xs, ys, Xt, yt, angles


def rotate_2d(X, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])  # Points are represented as rows
    return X @ R

Xs, ys, Xt, yt, angles = load_seq_two_moon_data(150, 150, 10,
                                                    max_angle=90, noise=0.1)
Xs1, ys1, Xt1, yt1, Xt_all, yt_all = load_battery_data()
# Xs2, ys2, Xt2, yt2, Xtest, ytest = load_mnist_data(dir_path='/Users/liuhanbing/Desktop/code/RotNIST/data/')
Xs, ys, Xt, yt, Xt_all, yt_all, acc, _, _, _, _ = load_battery_data_split(time_series=[10, 15, 20, 25, 30, 35, 40, 45, 50])
# print(Xtest[0].shape)

# print(ys.shape)
# print(Xt.shape)
# print(yt.shapes)
# pdb.set_trace()
# pl.figure()
# pl.plot(Xs[:, 0], Xs[:, 1], 'ob', label='Source samples')

# pl.plot(Xt[9][:, 0], Xt[9][:, 1], '^c', label='Target samples')

# pl.legend(loc=0)
# pl.show()

