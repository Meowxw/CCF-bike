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
import bikerental.online_evluation as oleva
import bikerental.lease_xgb_model as lease_xgb
import bikerental.return_xgb_model as return_xgb

# ----------------------------------------------------------------------------------------------------------------------#
def adjust_school(df_6799):

    submit_columns = df_6799.columns
    print(df_6799['LEASE'].sum()-df_6799['RT'].sum())

    df_6799.loc[:,'month'] = df_6799['time'].apply(lambda x:datetime.strptime(x,'%Y/%m/%d').month)
    df_6799.loc[:, 'dayint'] = df_6799['time'].apply(lambda x: np.int64(datetime.strftime(datetime.strptime(x, '%Y/%m/%d'),'%Y%m%d')))
    df_6799_month_mean = df_6799.groupby(['SHEDID','month'])['LEASE','RT'].median().reset_index()
    df_6799_month_mean.columns = ['SHEDID','month','lease_mean','return_mean']

    df_6799_month_mean9 = df_6799_month_mean[df_6799_month_mean.month==9]
    df_6799_month_mean9 = df_6799_month_mean9[['SHEDID','lease_mean','return_mean']]
    df_6799_month_mean9.columns = ['SHEDID','m9_lease_mean','m9_return_mean']

    df_6799_month_mean10 = df_6799_month_mean[df_6799_month_mean.month==10]
    df_6799_month_mean10 = df_6799_month_mean10[['SHEDID','lease_mean','return_mean']]
    df_6799_month_mean10.columns = ['SHEDID','m10_lease_mean','m10_return_mean']

    df_6799_month_mean10 = df_6799_month_mean[df_6799_month_mean.month == 10]
    df_6799_month_mean10 = df_6799_month_mean10[['SHEDID', 'lease_mean', 'return_mean']]
    df_6799_month_mean10.columns = ['SHEDID', 'm10_lease_mean', 'm10_return_mean']

    df_train_daily = util.get_daily_train()
    df_train_daily.loc[:,'month'] = df_train_daily['dayint'].apply(lambda x:datetime.strptime(str(x),'%Y%m%d').month)

    df_train_month_mean = df_train_daily.groupby(['SHEDID','month'])['lease_cnt','return_cnt'].median().reset_index()
    df_train_month_mean.columns = ['SHEDID','month','lease_mean','return_mean']

    df_train_month_mean5 = df_train_month_mean[df_train_month_mean.month==5]
    df_train_month_mean5 = df_train_month_mean5[['SHEDID','lease_mean','return_mean']]
    df_train_month_mean5.columns = ['SHEDID','m5_lease_mean','m5_return_mean']

    df_train_month_mean6 = df_train_month_mean[df_train_month_mean.month==6]
    df_train_month_mean6 = df_train_month_mean6[['SHEDID','lease_mean','return_mean']]
    df_train_month_mean6.columns = ['SHEDID','m6_lease_mean','m6_return_mean']

    df_train_mean = df_6799_month_mean9.merge(df_6799_month_mean10,on=['SHEDID'],how='left')
    df_train_mean = df_train_mean.merge(df_train_month_mean5,on=['SHEDID'],how='left')
    df_train_mean = df_train_mean.merge(df_train_month_mean6,on=['SHEDID'],how='left')

    df_6799 = df_6799.merge(df_train_mean, on='SHEDID', how='left')

    df_6799.loc[df_6799.LEASE < 0, 'LEASE'] = 0
    df_6799.loc[df_6799.RT < 0, 'RT'] = 0

    # 学校：将9和10月跟6和5月按中值对齐
    school_list = [58,78,79]

    df_6799.loc[(df_6799.month==9)&(df_6799.SHEDID.isin(school_list)),'LEASE'] \
                        = df_6799['LEASE']+df_6799['m6_lease_mean']-df_6799['m9_lease_mean']
    df_6799.loc[(df_6799.month==10)&(df_6799.SHEDID.isin(school_list)),'LEASE'] \
        = df_6799['LEASE']+df_6799['m5_lease_mean']-df_6799['m10_lease_mean']
    df_6799.loc[(df_6799.month==9)&(df_6799.SHEDID.isin(school_list)),'RT'] \
        = df_6799['RT']+df_6799['m6_return_mean']-df_6799['m9_return_mean']
    df_6799.loc[(df_6799.month==10)&(df_6799.SHEDID.isin(school_list)),'RT'] \
        = df_6799['RT']+df_6799['m5_return_mean']-df_6799['m10_return_mean']


    df_6799['LEASE'] = np.round(df_6799['LEASE'],2)
    df_6799['RT'] = np.round(df_6799['RT'],2)

    df_6799 = df_6799[submit_columns]
    df_6799.to_csv('../data/prediction/df_6799_modify_school.csv',index=False)

    print(df_6799['LEASE'].sum()-df_6799['RT'].sum())

    oleva.online_evaluation(df_6799)


# ----------------------------------------------------------------------------------------------------------------------#
# 主程序
if __name__ == '__main__':

    starttime = datetime.now()
    print('开始运行时间：{}'.format(starttime.strftime('%Y-%m-%d %H:%M:%S')))

    df_lease = lease_xgb.lease_xgb_model()
    df_return = return_xgb.return_xgb_model()

    df_lease = df_lease.merge(df_return,on=['SHEDID','time'],how='left')
    df_test = util.get_test_data()
    test_columns = df_test.columns

    df_test = df_test.merge(df_lease,on=['SHEDID','time'],how='left')
    df_test.loc[:, 'LEASE'] = df_test['lease_prediction']
    df_test.loc[:,'RT'] = df_test['return_prediction']
    df_test.loc[df_test.LEASE <= 0, 'LEASE'] = 1
    df_test.loc[df_test.RT <= 0, 'RT'] = 1
    df_test = df_test[test_columns]
    df_test.to_csv('../data/prediction/submission.csv',index=False)

    adjust_school(df_test)

    endtime = datetime.now()
    run_time_cnt = (endtime - starttime).seconds
    print('运行结束时间：{}，总运行时长：{}'.format(endtime.strftime('%Y-%m-%d %H:%M:%S'),run_time_cnt))

