#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: Anonymous
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import scipy as sc
import pandas as pd
import numpy as np
import math
import  measures
import time
import random
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

class featureSelector():
    
    def default(self,df):
        """
        By default, do nothing
        :param df:
        :return:
        """
        return df


    def remain_same(self,df):
        return df
    
    
    def _ent(self,data):
        """
        # Input a pandas series. calculate the entropy of series
        :param data:
        :return:
        """
        p_data = data.value_counts() / len(data)  # calculates the probabilities
        entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy
        return entropy
    
    def __init__(self):
        self.clf = None
        self.feature_importance = []
        self.train_X = None
        self.train_y = None
        
    def featureExtractor(self ,train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        self.clf = SelectKBest(chi2, k=12)
        self.train_X = self.clf.fit_transform(self.train_X, self.train_y)
        print("shape: after:", self.train_X)
        return self.train_X
    
    

    def gain_rank(self, df):
        """
        information gain attribute ranking
        reference: sect 2.1 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
        requires: discretization
        :param df:
        :return:
        """
        H_C = self._ent(df.iloc[:, -1])
        weights = pd.DataFrame(data=np.zeros([1, df.shape[1] - 1]), columns=df.columns[:-1])
    
        types_C = set(df.iloc[:, -1])
        target = df.columns[-1]
        for a_i, a in enumerate(df.columns[:-1]):  # for each attribute a
            for typea in set(df.loc[:, a]):  # each class of attribute a
                selected_a = df[df[a] == typea]
                sub = 0
                for typec in types_C:
                    p_c_a = selected_a[selected_a[target] == typec].shape[0] / selected_a.shape[0]
                    if p_c_a == 0:
                        continue
                    sub += p_c_a * math.log(p_c_a, 2)
                weights.loc[0, a] += -1 * selected_a.shape[0] / df.shape[0] * sub
    
        weights = H_C - weights
        weights[df.columns[-1]] = 1
        weights = weights.append([weights] * (df.shape[0] - 1), ignore_index=False)
        weights.index = df.index
        res = weights * df
        return res,weights.iloc[0].values[0:len(weights.iloc[0].values)-1]
    
    
    def relief(self, df, measures=measures.default):
        """
        reference: sect 2.2 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
        reference2: Kononenko et al. "Estimating Attributes: Analysis and Extensions of Relief"
        requires: discretization. distance measure provided
        :param measures:
        :param df:
        :return:
        """
        m = 20
        k = 10
        weights = pd.DataFrame(data=np.zeros([1, df.shape[1] - 1]), columns=df.columns[:-1])
        target = df.columns[-1]
    
        for i in range(m):
            selected_row = df.sample(1).iloc[0, :]
            dists = measures(selected_row, df)
            df['d_'] = dists
            hits = df[df[target] == df.iloc[0][-2]].iloc[:, :-1][:k] 
            miss = df[df[target] != df.iloc[0][-2]].iloc[:, :-1][:k]
#            print(hits)
#            import pdb
#            pdb.set_trace()
            t1 = np.sum(np.abs(hits.astype(np.float32) - selected_row.astype(np.float32)), axis=0) / (hits.shape[0] * m)
            t2 = np.sum(np.abs(miss.astype(np.float32) - selected_row.astype(np.float32)), axis=0) / (miss.shape[0] * m)
            
            weights = weights - t1 + t2
            df.drop(['d_'], axis=1, inplace=True)  # clear the distance
        weights = weights.drop(df.columns[-1], axis=1)
        weights = np.abs(weights)
        weights[df.columns[-1]] = 1
        weights = weights.append([weights] * (df.shape[0] - 1), ignore_index=True)
        weights.index = df.index
    
        return weights * df,weights.iloc[0].values
    
    def consistency_subset(self, df):
        """
        - Consistency-Based Subset Evaluation
        - Subset evaluator use Liu and Setino's consistency metric
        - reference: sect 2.5 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
    
        - requires: discreatization
        :param df:
        :return:
        """
        def consistency(sdf, classes):
            """
            Calculate the consistency of feature subset, which will be maximized
            :param sdf: dataframe regrading to a subset feature
            :return:
            """
            sdf = sdf.join(classes)
            uniques = sdf.drop_duplicates()
            target = classes.name
    
            subsum = 0
    
            for i in range(uniques.shape[0] - 1):
                row = uniques.iloc[i]
                matches = sdf[sdf == row].dropna()
                if matches.shape[0] <= 1: continue
                D = matches.shape[0]
                M = matches[matches[target] == float(matches.mode()[target])].shape[0]
                subsum += (D - M)
    
            return 1 - subsum / sdf.shape[0]
    
        features = df.columns[:-1]
        target = df.columns[-1]
    
        hc_starts_at = time.time()
        lst_improve_at = time.time()
        best = [0, None]
        while time.time() - lst_improve_at < 1 or time.time() - hc_starts_at < 5:
            # during of random_config search -> at most 5 seconds. if no improve by 1 second, then stop
            selects = [random.choice([0, 1]) for _ in range(len(features))]
            if not sum(selects): continue
            fs = [features[i] for i, v in enumerate(selects) if v]
            score = consistency(df[fs], df[target])
            if score > best[0]:
                best = [score, fs]
                lst_improve_at = time.time()
    
        selected_features = best[1] + [target]
        selected_features_list = []
        for feature in features:
            if feature in selected_features:
                selected_features_list.append(1)
            else:
                selected_features_list.append(0)
        return df[selected_features],selected_features_list
    
    
    def cfs(self,df):
        """
        - CFS = Correlation-based Feature Selection
        - reference: sect 2.4 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
        reference2: Hall et al. "Correlation-based Feature Selection for Discrete and Numeric Class Machine Learning"
        - Good feature subsets contain features highly corrleated with the calss, yet uncorrelated with each other.
        - random_config search is applied for figure out best feature subsets
        :param df:
        :return:
        """
    
        features = df.columns[:-1]
        target = df.columns[-1]
        cf = pd.DataFrame(data=np.zeros([1, df.shape[1] - 1]), columns=features, index=df.columns[-1:])
        ff = pd.DataFrame(data=np.zeros([len(features), len(features)]), index=features, columns=features)
        # fill in cf
        for attr in cf.columns:
            cf.loc[target, attr] = abs(df[attr].corr(df[target], method='pearson'))
        
    
        # fill in ff
        for attr1 in ff.index:
            for attr2 in ff.columns:
                if attr1 == attr2: continue
                if ff.loc[attr1, attr2]: continue
                corr = abs(df[attr1].corr(df[attr2], method='pearson'))
                ff.loc[attr1, attr2] = corr
                ff.loc[attr2, attr1] = corr
    
        def merit_S(fs, cf, ff):
            """
            Calculate the heuristic (to maximize) according to Ghiselli 1964. eq1 in ref2
            :param ff:
            :param cf:
            :param fs: feature_subset names
            :return:
            """
            r_cf = cf[fs].mean().mean()
            r_ff = ff.loc[fs, fs].mean().mean()
            k = len(fs)
            return k * r_cf / math.sqrt(k + (k - 1) * r_ff)
    
        # use stochastic search algorithm to figure out best subsets
        # features subsets are encoded as [0/1]^F
    
        hc_starts_at = time.time()
        lst_improve_at = time.time()
        best = [0, None]
        while time.time() - lst_improve_at < 1 or time.time() - hc_starts_at < 5:
            # during of random_config search -> at most 5 seconds. if no improve by 1 second, then stop
            selects = [random.choice([0, 1]) for _ in range(len(features))]
            if not sum(selects): continue
            fs = [features[i] for i, v in enumerate(selects) if v]
            score = merit_S(fs, cf, ff)
            if score > best[0]:
                best = [score, fs]
                lst_improve_at = time.time()
    
        selected_features = best[1] + [target]
        selected_features_list = []
        for feature in features:
            if feature in selected_features:
                selected_features_list.append(1)
            else:
                selected_features_list.append(0)
        return df[selected_features],selected_features_list,selected_features

    
    def cfs_bfs(self,df):
        """
        - CFS = Correlation-based Feature Selection
        - reference: sect 2.4 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
        reference2: Hall et al. "Correlation-based Feature Selection for Discrete and Numeric Class Machine Learning"
        - Good feature subsets contain features highly corrleated with the calss, yet uncorrelated with each other.
        - random_config search is applied for figure out best feature subsets
        :param df:
        :return:
        """

        print("{")
    
        features = df.columns[:-1]
        target = df.columns[-1]
        print(target)
        cf = pd.DataFrame(data=np.zeros([1, df.shape[1] - 1]), columns=features, index=df.columns[-1:])
        ff = pd.DataFrame(data=np.zeros([len(features), len(features)]), index=features, columns=features)

        # fill in cf
        for attr in cf.columns:
            cf.loc[target, attr] = abs(df[attr].corr(df[target], method='pearson'))
        # fill in ff
        for attr1 in ff.index:
            for attr2 in ff.columns:
                if attr1 == attr2: continue
                if ff.loc[attr1, attr2]: continue
                corr = abs(df[attr1].corr(df[attr2], method='pearson'))
                ff.loc[attr1, attr2] = corr
                ff.loc[attr2, attr1] = corr
    
        def merit_S(fs, cf, ff):
            """
            Calculate the heuristic (to maximize) according to Ghiselli 1964. eq1 in ref2
            :param ff:
            :param cf:
            :param fs: feature_subset names
            :return:
            """
            r_cf = cf[fs].mean().mean()
            r_ff = ff.loc[fs, fs].mean().mean()
            k = len(fs)
            return round(k * r_cf / math.sqrt(k + (k - 1) * r_ff),2)
    
        # use stochastic search algorithm to figure out best subsets
        # features subsets are encoded as [0/1]^F
    
        F = []
        # M stores the merit values
        M = []
        while True:
            score = -100000000000
            idx = -1
            for i in features:
                if i not in F:
                    F.append(i)
                    # calculate the merit of current selected features
                    t = merit_S(F,cf,ff)
                    if t > score:
                        score = t
                        idx = i
                    F.pop()
            F.append(idx)
            M.append(score)
            similarity = 0
            best = max(M)
            if len(M) > 5:
                if score <= M[len(M)-2]:
                    similarity += 1
                    if score <= M[len(M)-3]:
                        similarity += 1
                        if score <= M[len(M)-4]:
                            similarity += 1
                            if score <= M[len(M)-5]:
                                similarity += 1
                                break 
        print(F,M)                      
        F = F[0:len(M)-similarity]
        selected_features = F + [target]
        selected_features_list = []
        for feature in features:
            if feature in selected_features:
                selected_features_list.append(1)
            else:
                selected_features_list.append(0)

        print("}")
        return df[selected_features],selected_features_list,selected_features


    def tfs(self,df,n_estimators=50):
        """
            - tfs = Tree-based feature selection
            - reference: 
            - Tree-based estimators (see the sklearn.tree module and forest of trees in the sklearn.ensemble module)
            used to compute feature importances.
            :param df:
            :return:
        """
        target = df.columns[-1]
        X = df.drop(labels = [target], axis=1)
        y = df[target]
        clf = ExtraTreesClassifier(n_estimators=n_estimators)
        clf.fit(X,y)
        return 0,clf.feature_importances_

    
    def l1(self,df,C=0.01,dual=False):
        """
            - tfs = l1 regularization based feature selector
            - reference: 
            - Tree-based estimators (see the sklearn.tree module and forest of trees in the sklearn.ensemble module)
            used to compute feature importances.
            :param df:
            :return:
        """
        features = df.columns[:-1]
        target = df.columns[-1]
        X = df.drop(labels = [target], axis=1)
        y = df[target]
        clf = LinearSVC(C=C, penalty="l1", dual=dual)
        clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        selected_features = model.get_support(indices=False)
        selected_features_list = []
        for i in range(len(features)):
            if selected_features[i] == True:
                selected_features_list.append(1)
            else:
                selected_features_list.append(0)
        return 0,selected_features_list
