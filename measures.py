
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:24:45 2018

@author: Anonymous
"""




from __future__ import division
import numpy as np
import pandas as pd
import hashlib
import pdb


"""
input:
 row - one pandas.core.series.Series. INCLUDING LAST Y VALUE
 train - pandas.dataframe
output: pandas.core.series.Series.
"""


def default(row, train):
    return euclidean(row, train)


def euclidean(row, train):
    # ignoring the last y value first
    row = row[:-1]
    train = train.iloc[:, :-1]
    return np.sum(np.square(row - train), axis=1) ** 0.5


cache = dict()


def weighted_euclidean(row, train):
    # get the weight according to paper "A Comparative Study of
    # Cost Estimation Models for Web Hypermedia Applications"

    # to avoid repeat computation, check for hash first
    tid = hashlib.sha256(train.values.tobytes()).hexdigest()
    if tid in cache:
        weight = cache[tid]
    else:
        weight = np.array([])
        effect = train.iloc[:, -1]
        for col in range(train.shape[1] - 1):
            alpha = train.iloc[:, col].corr(effect, method='pearson')
            if abs(alpha) > 0.01:
                alpha = 2
            else:
                alpha = 1
            weight = np.append(weight, alpha)
        cache[tid] = weight

    # ignoring the last y value
    row = row[:-1]
    train = train.iloc[:, :-1]

    return np.sum((row - train) ** 2 * weight, axis=1) ** 0.5


def maximum_measure(row, train):
    # ignoring the last y value first
    row = row[:-1]
    train = train.iloc[:, :-1]
    return np.max(np.square(row - train), axis=1)


def local_likelihood(row, train, k=-1):
    # if k = -1, than use default value, i.e. len(train)*0.2
    # references frank et al. "Locally Weighted Naive Bayes"
    # requiring normalization
    if k == -1:
        k = int(train.shape[0] * 0.2)
    res = euclidean(row, train)
    threshold = sorted(res.tolist())[k]
    res[res >= threshold] = 0
    return res


def minkowski(row, train, p=2):
    # ignoring the last y value first
    row = row[:-1]
    train = train.iloc[:, :-1]
    return np.sum((row - train) ** p, axis=1) ** (1 / p)


def feature_mean_dist(row, train):
    # ignoring the last y value first
    row = row[:-1]
    train = train.iloc[:, :-1]
    return pd.Series(data=np.average(row) - np.average(train, axis=1))