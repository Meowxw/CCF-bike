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

import utility as util


# ----------------------------------------------------------------------------------------------------------------------#
# SHEDID,lease_time,dayint,afternoon,lease_cnt
# 上个月同期的借车次数
# 上个月最后1/3/7/14/正月的平均借车次数
def get_lease_feature(col_name):
    df_train_label = util.get_daily_train()

    df_holiday = util.get_holiday_features()
    df_train_label = df_train_label.merge(df_holiday, on='dayint', how='left')
    print('df_train_label length:{} after get_holiday_features'.format(len(df_train_label)))

    df_train_label.loc[:, 'dt'] = df_train_label['dayint'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    df_train_label.loc[:, 'month'] = df_train_label['dt'].dt.month
    df_train_label.loc[:, 'day'] = df_train_label['dt'].dt.day
    df_train_label.loc[:, 'dayofmonth'] = df_train_label['dt'].apply(lambda x: calendar.monthrange(x.year, x.month)[1])

    # 要做时间窗特征，所以需要扩展为时间窗对比
    df_train_label_win = df_train_label.copy()
    df_train_label_win.loc[:,'join_key'] = 1
    df_month_window = DataFrame({'win_month':np.arange(1,df_train_label.month.max()+2),'join_key':1})
    #print('befor join month_window :{}'.format(len(df_train_label)))
    df_train_label_win = df_train_label_win.merge(df_month_window,on='join_key',how='left')
    #print('after join month_window :{}'.format(len(df_train_label)))

    # 上个月最后一周的特征
    df_lastweek = df_train_label[df_train_label.day >= (df_train_label.dayofmonth - 6)]
    df_lastweek = df_lastweek.groupby(['SHEDID', 'month'])[col_name+'_cnt'].agg({
        'lastweek_min_' + col_name+'_cnt': 'min'
        , 'lastweek_max_' + col_name+'_cnt': 'max'
        , 'lastweek_mean_' + col_name+'_cnt': 'mean'
        , 'lastweek_median_' + col_name+'_cnt': 'median'
        , 'lastweek_var_' + col_name+'_cnt': 'var'
        , 'lastweek_std_' + col_name+'_cnt': 'std'
        ,'lastweek_quantile10_' + col_name+'_cnt': lambda x: x.quantile(0.1)
        ,'lastweek_quantile20_' + col_name+'_cnt': lambda x: x.quantile(0.2)
        ,'lastweek_quantile30_' + col_name+'_cnt': lambda x: x.quantile(0.3)
        ,'lastweek_quantile40_' + col_name+'_cnt': lambda x: x.quantile(0.4)
        ,'lastweek_quantile60_' + col_name+'_cnt': lambda x: x.quantile(0.6)
        ,'lastweek_quantile70_' + col_name+'_cnt': lambda x: x.quantile(0.7)
        ,'lastweek_quantile80_' + col_name+'_cnt': lambda x: x.quantile(0.8)
        ,'lastweek_quantile90_' + col_name+'_cnt': lambda x: x.quantile(0.9)
        ,'lastweek_mad_' + col_name+'_cnt': 'mad'
        ,'lastweek_sem_' + col_name+'_cnt': 'sem'
        ,'lastweek_kurt_' + col_name+'_cnt': lambda x: x.kurt()
        ,'lastweek_skew_' + col_name+'_cnt': 'skew'
    }).reset_index()
    df_lastweek = df_lastweek[(df_lastweek.month >= 2) & (df_lastweek.month <= 8)]
    df_lastweek.loc[:, 'month'] = df_lastweek['month'] + 1  # 上一个同期作为下个月的参考
    df_lastweek10 = df_lastweek[df_lastweek.month == 9]
    df_lastweek10.loc[:, 'month'] = df_lastweek10['month'] + 1
    df_lastweek = pd.concat([df_lastweek, df_lastweek10], ignore_index=True)

    # 上个月最后周几的均值等
    #print(df_train_label.dtypes)
    df_lastweekday = df_train_label.groupby(['SHEDID', 'month', 'weekday'])[col_name+'_cnt'].agg({
        'lastweekday_min_' + col_name+'_cnt': 'min'
        , 'lastweekday_max_' + col_name+'_cnt': 'max'
        , 'lastweekday_mean_' + col_name+'_cnt': 'mean'
        , 'lastweekday_median_' + col_name+'_cnt': 'median'
        , 'lastweekday_var_' + col_name+'_cnt': 'var'
        , 'lastweekday_std_' + col_name+'_cnt': 'std'
        , 'lastweekday_quantile10_' + col_name: lambda x: x.quantile(0.1)
        , 'lastweekday_quantile20_' + col_name: lambda x: x.quantile(0.2)
        , 'lastweekday_quantile30_' + col_name: lambda x: x.quantile(0.3)
        , 'lastweekday_quantile40_' + col_name: lambda x: x.quantile(0.4)
        , 'lastweekday_quantile60_' + col_name: lambda x: x.quantile(0.6)
        , 'lastweekday_quantile70_' + col_name: lambda x: x.quantile(0.7)
        , 'lastweekday_quantile80_' + col_name: lambda x: x.quantile(0.8)
        , 'lastweekday_quantile90_' + col_name: lambda x: x.quantile(0.9)
        , 'lastweekday_mad_' + col_name: 'mad'
        , 'lastweekday_sem_' + col_name: 'sem'
        , 'lastweekday_kurt_' + col_name: lambda x: x.kurt()
        , 'lastweekday_skew_' + col_name: 'skew'
    }).reset_index()
    df_lastweekday = df_lastweekday[(df_lastweekday.month >= 2) & (df_lastweekday.month <= 8)]
    df_lastweekday.loc[:, 'month'] = df_lastweekday['month'] + 1  # 上一个同期作为下个月的参考
    df_lastweekday10 = df_lastweekday[df_lastweekday.month == 9]
    df_lastweekday10.loc[:, 'month'] = df_lastweekday10['month'] + 1
    df_lastweekday = pd.concat([df_lastweekday, df_lastweekday10], ignore_index=True)

    df_train_label_win = df_train_label_win[df_train_label_win.month<df_train_label_win.win_month]
    df_hisweekday = df_train_label_win.groupby(['SHEDID', 'win_month', 'weekday'])[col_name + '_cnt'].agg({
        'hisweekday_min_' + col_name + '_cnt': 'min'
        , 'hisweekday_max_' + col_name + '_cnt': 'max'
        , 'hisweekday_mean_' + col_name + '_cnt': 'mean'
        , 'hisweekday_median_' + col_name + '_cnt': 'median'
        , 'hisweekday_var_' + col_name + '_cnt': 'var'
        , 'hisweekday_std_' + col_name + '_cnt': 'std'
        , 'hisweekday_quantile10_' + col_name: lambda x: x.quantile(0.1)
        , 'hisweekday_quantile20_' + col_name: lambda x: x.quantile(0.2)
        , 'hisweekday_quantile30_' + col_name: lambda x: x.quantile(0.3)
        , 'hisweekday_quantile40_' + col_name: lambda x: x.quantile(0.4)
        , 'hisweekday_quantile60_' + col_name: lambda x: x.quantile(0.6)
        , 'hisweekday_quantile70_' + col_name: lambda x: x.quantile(0.7)
        , 'hisweekday_quantile80_' + col_name: lambda x: x.quantile(0.8)
        , 'hisweekday_quantile90_' + col_name: lambda x: x.quantile(0.9)
        , 'hisweekday_mad_' + col_name: 'mad'
        , 'hisweekday_sem_' + col_name: 'sem'
        , 'hisweekday_kurt_' + col_name: lambda x: x.kurt()
        , 'hisweekday_skew_' + col_name: 'skew'
    }).reset_index()
    df_hisweekday.rename(columns={'win_month':'month'},inplace=True)
    df_hisweekday = df_hisweekday[(df_hisweekday.month >= 2) & (df_hisweekday.month <= 8)]
    df_hisweekday.loc[:, 'month'] = df_hisweekday['month'] + 1  # 上一个同期作为下个月的参考
    df_hisweekday10 = df_hisweekday[df_hisweekday.month == 9]
    df_hisweekday10.loc[:, 'month'] = df_hisweekday10['month'] + 1
    df_hisweekday = pd.concat([df_hisweekday, df_hisweekday10], ignore_index=True)

    # 上个月的骑行数据统计：
    # # 变量名           标签
    # LEASEDATE     借车日期
    # LEASETIME     借车时间
    # USETIME       骑行分钟
    # OVERTIME      超时分钟
    # SHEDID        借车站点号
    # LOCATIVEID    车位
    # CARDSN        车卡
    # VIPCARDSN     借车卡
    # RTSHEDID      还车站点号
    # RTLOCATIVEID  还车车位
    # RTDATE        还车日期
    # RTTIME        还车时间
    # OPTYPE        操作类型
    # OPNAME        操作名称

    df_train = util.get_train_data()
    df_train.loc[:,'month'] = df_train['lease_dt'].dt.month
    df_train.loc[:,'weekday'] = df_train['lease_dt'].dt.dayofweek + 1
    print(df_train.weekday.unique())

    # non-using days of each month
    df_train_non_using = df_train_label[df_train_label.month <= 8]
    df_nonusing = df_train_non_using[df_train_non_using[col_name+'_cnt'] == 0]
    df_nonusing_stat = df_nonusing.groupby(['SHEDID', 'month'])[col_name+'_cnt'].count().reset_index()
    df_nonusing_stat.columns = ['SHEDID', 'month', 'non_using_day_count']
    # df_nonusing_stat.to_csv('df_nonusing_stat.csv',index=False)

    df_test = util.get_test_data()
    df_ext_time = util.get_ext_time()
    df_ext_time.columns = ['SHEDID', 'time', 'dayint']
    df_ext_time = df_ext_time[df_ext_time.dayint >= 20150901]

    df_ext_time = df_ext_time.merge(df_test, on=['SHEDID', 'time'], how='left')
    df_ext_time = df_ext_time[df_ext_time.RT.isnull()]
    # df_ext_time.sort_values(by=['SHEDID','dayint'],ascending=True,inplace=True)
    # print(df_ext_time)
    df_ext_time.loc[:, 'month'] = df_ext_time['dayint'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').month)
    # print(df_ext_time.head(10))
    df_test_stat = df_ext_time.groupby(['SHEDID', 'month'])['time'].count().reset_index()
    df_test_stat.columns = ['SHEDID', 'month', 'non_using_day_count']
    # print(df_test_stat)
    # df_test_stat.to_csv('df_test_stat.csv', index=False)

    df_nonusing_stat = pd.concat([df_nonusing_stat, df_test_stat], ignore_index=True)
    # df_nonusing_stat.to_csv('df_nonusing_stat_merged.csv',index=False)

    # shedid statistic features
    df_train = util.get_train_data()
    df_train.loc[:, 'month'] = df_train['lease_dt'].dt.month
    # <SHEDID,lease_time>，接下来是特征提取
    # 不同SHEDID的车位数
    df_lease_locativeid = df_train[['SHEDID', 'month', 'LOCATIVEID']].drop_duplicates()
    df_return_locativeid = df_train[['RTSHEDID', 'month', 'RTLOCATIVEID']].drop_duplicates()
    df_return_locativeid.columns = ['SHEDID', 'month', 'LOCATIVEID']
    df_locativeid = pd.concat([df_lease_locativeid, df_return_locativeid], ignore_index=True).drop_duplicates()

    df_locativeid_stat = df_locativeid.groupby(['SHEDID', 'month'])['LOCATIVEID']. \
        agg({
        'shed_locative_used_count': 'count'
        , 'shed_locative_build_count': 'max'
    }).reset_index()
    df_locativeid_stat.loc[:, 'shed_locative_active_rate'] = df_locativeid_stat['shed_locative_used_count'] / \
                                                             df_locativeid_stat['shed_locative_build_count']
    df_locativeid_stat = df_locativeid_stat[(df_locativeid_stat.month >= 2) & (df_locativeid_stat.month <= 8)]
    df_locativeid_stat.loc[:, 'month'] = df_locativeid_stat['month'] + 1  # 上一个同期作为下个月的参考
    df_locativeid_stat10 = df_locativeid_stat[df_locativeid_stat.month == 9]
    df_locativeid_stat10.loc[:, 'month'] = df_locativeid_stat10['month'] + 1
    df_locativeid_stat = pd.concat([df_locativeid_stat, df_locativeid_stat10], ignore_index=True)

    df_train_label = df_train_label.merge(df_locativeid_stat, on=['SHEDID', 'month'], how='left')
    df_train_label.fillna(-999, inplace=True)

    # 合并后返回
    df_train_label = df_train_label.merge(df_nonusing_stat, on=['SHEDID', 'month'], how='left')
    df_train_label.fillna(-999, inplace=True)


    df_train_label = df_train_label.merge(df_lastweek, on=['SHEDID', 'month'], how='left')
    df_train_label = df_train_label.merge(df_lastweekday, on=['SHEDID', 'month', 'weekday'], how='left')
    print('df_train_label length:{} after df_lastweekday'.format(len(df_train_label)))
    df_train_label = df_train_label.merge(df_hisweekday, on=['SHEDID', 'month', 'weekday'], how='left')
    print('df_train_label length:{} after df_hisweekday'.format(len(df_train_label)))
    #df_train_label = df_train_label.merge(df_lastweekday_usetime, on=['SHEDID', 'month', 'weekday'], how='left')
    #print('df_train_label length:{} after df_lastweekday_usetime'.format(len(df_train_label)))
    #df_train_label = df_train_label.merge(df_1min_stat, on=['SHEDID', 'month'], how='left')
    #print('df_train_label length:{} after df_1min_stat'.format(len(df_train_label)))

    df_position = util.get_shed_position()
    df_train_label = df_train_label.merge(df_position, on=['SHEDID'], how='left')

    del df_train_label['dt']

    return df_train_label


# ----------------------------------------------------------------------------------------------------------------------#
# 201505/06/07预测201508
# <SHEDID,lease_time,lease_cnt>
def lease_xgb_model():


    df_train_label = get_lease_feature('lease')
    print('df_train_label length:{} after get_lease_feature'.format(len(df_train_label)))
    print(df_train_label.columns)

    # date time related features
    df_tianqi = util.get_weather()
    df_train_label = df_train_label.merge(df_tianqi, on='dayint', how='left')
    # df_train_label.fillna(-999,inplace=True)
    print('df_train_label length:{} after get_weather'.format(len(df_train_label)))

    df_train_label.loc[:, 'SHEDID1'] = df_train_label['SHEDID']
    df_train_label = pd.get_dummies(df_train_label, columns=['SHEDID1'])

    params = {
        'objective': 'reg:linear',
        'eta': 0.001,
        'max_depth': 7,
        'eval_metric': 'rmse',
        'seed': 518,
        'colsample': 0.8,
        'subsample': 0.8,
        'missing': -999,
        'silent':1
    }

    feature = [x for x in df_train_label.columns if
               x not in ['SHEDID', 'time', 'dayint', 'lease_cnt','return_cnt','month']]
    feature.sort()
    print(feature)


    df_train = df_train_label[(df_train_label.dayint <= 20150831) & (df_train_label.dayint >= 20150301)]
    df_test = df_train_label[df_train_label.dayint >= 20150901]

    X_train, X_test, y_train, y_test = train_test_split(df_train[feature], df_train['lease_cnt'],test_size=0.33, random_state=518)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    # ---------------- xgboost model -----------------------#
    print('X_train length:{},X_test length:{}.'.format(len(X_train), len(X_test)))
    dtrain = xgb.DMatrix(X_train[feature], y_train)
    dvalid = xgb.DMatrix(X_test[feature], y_test)

    xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [(dtrain, 'train'), (dvalid, 'test')]
    num_rounds = 200000
    model = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=15,verbose_eval=1000)
    
    df_test.loc[:, 'lease_prediction'] = np.int64(np.round(model.predict(xgbtest), 0))
    df_test.to_csv('../data/prediction/xgb_lease_prediction.csv', index=False)

    df_result = df_test[['SHEDID', 'time', 'lease_prediction']]
    df_result.to_csv('../data/prediction/xgb_lease_submission.csv', index=False)

    return df_result

# ----------------------------------------------------------------------------------------------------------------------#
# 主程序
if __name__ == '__main__':

    #data_exploration()

    #df_weather = get_weather()
    #print(df_weather)


    starttime = datetime.now()
    print('开始运行时间：{}'.format(starttime.strftime('%Y-%m-%d %H:%M:%S')))

    df_lease = lease_xgb_model()

    df_test = util.get_test_data()
    test_columns = df_test.columns

    df_test = df_test.merge(df_lease,on=['SHEDID','time'],how='left')
    df_test.loc[:, 'LEASE'] = df_test['lease_prediction']

    df_test.loc[df_test.LEASE < 0, 'LEASE'] = 0
    df_test.loc[df_test.RT < 0, 'RT'] = 0
    df_test = df_test[test_columns]
    df_test.to_csv('../data/prediction/lease_submission.csv',index=False)

    endtime = datetime.now()
    run_time_cnt = (endtime - starttime).seconds
    print('运行结束时间：{}，总运行时长：{}'.format(endtime.strftime('%Y-%m-%d %H:%M:%S'),run_time_cnt))

