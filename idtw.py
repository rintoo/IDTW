import numpy as np
import xgboost as xgb
import pandas as pd
from joblib import Parallel, delayed
import tensorflow as tf
import random
import nevergrad as ng
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from concurrent import futures
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from scipy import special
import sys
import warnings
import scipy.stats as st
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
inf = float("inf")

def cal_dtw_distance(ts_a, ts_b, direc = 'ud',window=None):
    d = lambda x, y: (x - y)**2
    max_warping_window = 10000
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    r, c = len(ts_a), len(ts_b)
    x = np.linspace(0, 1, max(r,c), endpoint=False)
    y = st.multivariate_normal.pdf(x, mean=0, cov=.5)
    y = y / np.max(y)
    cost = np.full((r + 1, c + 1), inf)
    if window is None:
        window = max(r, c)
    cost[0, 0] = d(ts_a[0], ts_b[0])#d[0,0]
    # for i in range(1, r):
    #     cost[i, 0] = 0
    # for j in range(1, c):
    #     cost[0, j] = 0
    num = 0
    if direc == 'lr':
        for i in range(1, r):
            for j in range(max(1, i - max(0, r - c) - window + 1),
                           min(c, i + max(0, c - r) + window)):
                choices = [cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]]
                direction = np.argmin(choices)
                choices = np.array([cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]])
                if (direction == 1):#direction != 0
                    if (num != 0)&(cost[i - 1, j - 1] != inf)&(cost[i, j - 1] != inf):
                        alpha = num+1
                        theta = cost[i, j - 1]**(1/alpha) / (cost[i - 1, j - 1]**(1/alpha)+ (cost[i - 1, j - 1]-cost[i, j - 1])**(1/alpha))
                        cost[i, j - 1] = (1+theta)*cost[i, j - 1]
                    num += 1
                else:
                    num = 0
                decay = y[-abs(i - j)]
                cost[i, j] = min(choices) + decay * d(ts_a[i], ts_b[j])
    elif direc == 'ud':
        for j in range(1,c):
            for i in range(max(1, j - max(0, c - r) - window + 1),
                           min(r, j + max(0, r - c) + window)):
                choices = [cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]]
                direction = np.argmin(choices)
                choices = np.array([cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]])
                if (direction == 2):#direction == 2
                    if (num != 0)&(cost[i - 1, j - 1] != inf)&(cost[i - 1, j] != inf):
                        alpha = num+1
                        theta = cost[i - 1, j]**(1/alpha) / (cost[i - 1, j - 1]**(1/alpha) + (cost[i - 1, j - 1]-cost[i - 1, j])**(1/alpha))
                        cost[i - 1, j] = (1 + theta) * cost[i - 1, j]
                    num += 1
                else:
                    num = 0
                decay = y[-abs(i-j)]
                cost[i, j] = min(choices) + decay*d(ts_a[i], ts_b[j])#d[i,j]
    cost = cost[:-1,:-1]
    return cost[-1, -1],cost


def warping_path(from_s, to_s, **kwargs):
    dist_ud, paths_ud = cal_dtw_distance(from_s, to_s,direc='ud')
    dist_lr, paths_lr = cal_dtw_distance(from_s, to_s,direc='lr')
    paths = paths_ud*paths_lr
    return paths[-1,-1]


def best_path(s1, s2,paths, row=None, col=None, use_max=False):
    if use_max:
        argm = np.argmax
    else:
        argm = np.argmin
    if row is None:
        i = int(paths.shape[0] - 1)
    else:
        i = row
    if col is None:
        j = int(paths.shape[1] - 1)
    else:
        j = col
    p = []
    cost = []
    if paths[i, j] != -1:
        p.append((i - 1, j - 1))
    while i > 0 and j > 0:
        temp = [paths[i - 1, j - 1], paths[i - 1, j], paths[i, j - 1]]
        c = argm(temp)
        if c == 0:
            i, j = i - 1, j - 1
        elif c == 1:
            i = i - 1
        elif c == 2:
            j = j - 1

        if paths[i, j] != -1:
            p.append((i - 1, j - 1))
            try:
                cost.append((s1[i-1] - s2[j-1]) ** 2)
            except:
                pass
    p.pop()
    p.reverse()
    V = np.var(cost)
    return p,V

df_name = pd.read_excel('../DataSummary.xlsx',sheet_name='len250',header=0)#选择长度250以下避免算太久
RATE = []
name_temple = []
def dtw_handle(i):
    global df_name
    global RATE
    global name_temple
    name = df_name.iloc[i, 0]
    num_class = df_name.iloc[i, 1]
    df = pd.read_csv('../UCRArchive_2018/' + name + '/' + name + '_TRAIN.tsv', sep='\t',
                     header=None)
    df_test = pd.read_csv('../UCRArchive_2018/' + name + '/' + name + '_TEST.tsv', sep='\t', header=None)
    X_train = df.iloc[:, 1:]
    y_train = df.iloc[:, 0]
    X_test = df_test.iloc[:, 1:]
    y_test = df_test.iloc[:, 0]
    dist = []
    knn = KNeighborsClassifier(n_neighbors=1,
                               algorithm='auto',
                               metric=warping_path)
    knn.fit(X_train, y_train)
    Y_hat = knn.predict(X_test)
    result = pd.DataFrame(y_test == Y_hat).value_counts()
    rate = result[0] / len(df_test)
    print(name + "'s KNN fit done!")
    RATE.append(rate)
    name_temple.append(name)
    np.savetxt('result_name.csv', np.array(name_temple), delimiter=',')
    np.savetxt('result.csv', np.array(RATE), delimiter=',')
Parallel(n_jobs=-1, backend='loky')(delayed(dtw_handle)(i) for i in range(len(df_name)))