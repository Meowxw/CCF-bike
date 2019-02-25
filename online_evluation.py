import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold,train_test_split, cross_val_score
import calendar
import os
import glob
import matplotlib.pyplot as plt
from scipy import sparse
from datetime import *
import time
import pickle
import xgboost as xgb
import xgbfir

import bikerental.utility as util


# ----------------------------------------------------------------------------------------------------------------------#
def online_evaluation(df_submission):
    path = '../data/submission/offline_online_evaluation/'

    score_list = []
    parents = os.listdir(path)

    df_lease = df_submission[['SHEDID','time','LEASE']]
    df_return = df_submission[['SHEDID', 'time', 'RT']]

    for parent in parents:
        child = os.path.join(path, parent)

        score = child.split('.')[-2]
        score = score[1:]
        score_list.append(score)

        df = pd.read_csv(child)
        df.columns = ['SHEDID','time','RT_'+score,'LEASE_'+score]

        df_lease1 = df[['SHEDID','time','LEASE_'+score]]
        df_return1 = df[['SHEDID', 'time', 'RT_' + score]]

        df_lease = df_lease.merge(df_lease1,on=['SHEDID','time'],how='left')
        df_return = df_return.merge(df_return1, on=['SHEDID', 'time'], how='left')

    #  calculating the relationship
    lease_columns = []
    list_lease_score = []
    list_lease_diff = []
    list_lease_abs_diff = []
    for score in score_list :
        lease_columns.append(score)
        list_lease_score.append(1/(1+np.sqrt(((df_lease['LEASE_'+score] - df_lease['LEASE'])**2).sum()/len(df_return))))
        list_lease_diff.append((df_lease['LEASE_'+score] - df_lease['LEASE']).sum()/len(df_return))
        list_lease_abs_diff.append(np.abs(df_lease['LEASE_' + score] - df_lease['LEASE']).sum() / len(df_return))
    #print(list_lease_score)

    df_lease_score = DataFrame({'his_file_score':lease_columns,'lease_score':list_lease_score,'lease_diff':list_lease_diff,'lease_abs_diff':list_lease_abs_diff})

    return_columns = []
    list_return_score = []
    list_return_diff = []
    list_return_abs_diff = []
    for score in score_list:
        return_columns.append(score)
        list_return_score.append(1/(1+np.sqrt(((df_return['RT_' + score] - df_return['RT']) ** 2).sum()/len(df_return))))
        list_return_diff.append((df_return['RT_' + score] - df_return['RT']).sum()/len(df_return))
        list_return_abs_diff.append(np.abs(df_return['RT_' + score] - df_return['RT']).sum() / len(df_return))
    # print(list_lease_score)

    df_return_score = DataFrame({'his_file_score': return_columns, 'return_score': list_return_score, 'return_diff': list_return_diff, 'return_abs_diff': list_return_abs_diff})
    df_lease_score = df_lease_score.merge(df_return_score,on=['his_file_score'])
    print(df_lease_score)

    return df_lease_score


# ----------------------------------------------------------------------------------------------------------------------#
# 主程序
if __name__ == '__main__':
    file_name = '../data/prediction/submission.csv'

    df_submission = pd.read_csv(file_name)
    online_evaluation(df_submission)




