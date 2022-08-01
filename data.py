import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
import matplotlib.pylab as pl
import pdb
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

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
    dir = '/Users/liuhanbing/Desktop/code/out_SOC_005-075_excel/'
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
    dir = '/Users/liuhanbing/Desktop/code/out_SOC_005-075_excel/'
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
Xs2, ys2, Xt2, yt2, Xtest, ytest = load_mnist_data(dir_path='/Users/liuhanbing/Desktop/code/RotNIST/data/')
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

