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

# ----------------------------------------------------------------------------------------------------------------------#
def get_test_data():
    df_test = pd.read_csv('../data/input/s2_test.csv')
    return df_test

# ----------------------------------------------------------------------------------------------------------------------#
OPNAME_TO_CODE = {
    '会员卡借车':1,
    '会员卡还车':2,
    '还车故障':3,
    '无卡还车':4 }

# ----------------------------------------------------------------------------------------------------------------------#
# 借还行为表　
# 变量名           标签
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

# 无卡还车：承租人将车辆还入桩位时计费截止，若没有进行还车刷卡确认操作，此次的租车费用将在下次刷卡时扣除。下一次该承租人刷卡时将会出现“还车成功”的语音提示，这会给人产生疑惑。因此，建议市民还车时刷卡予以确认，以免产生不必要的困扰。

def get_train_data():
    file_name = '../data/middle/bike_train.pkl'
    if not os.path.exists(file_name):
        df_train = pd.read_csv('../data/input/s2_train.csv'
                               #,nrows=100
                                ,encoding='GB2312')

        df_train.loc[:,'lease_dt'] = df_train.apply(lambda x:datetime.strptime(x['LEASEDATE']+' '+x['LEASETIME'],"%m/%d/%Y %H:%M:%S"),axis=1)
        df_train.loc[:, 'return_dt'] = df_train.apply(lambda x: datetime.strptime(x['RTDATE'] + ' ' + x['RTTIME'], "%m/%d/%Y %H:%M:%S"), axis=1)

        del df_train['LEASEDATE']
        del df_train['LEASETIME']
        del df_train['RTDATE']
        del df_train['RTTIME']

        # 借车时间
        df_train.loc[:,'lease_time'] = df_train['lease_dt'].apply(lambda x:('{}/{}/{}'.format(x.year,x.month,x.day)))
        df_train.loc[:, 'return_time'] = df_train['return_dt'].apply(lambda x:('{}/{}/{}'.format(x.year,x.month,x.day)))

        # 车辆使用时长
        df_train.loc[:, 'same_shedid'] = np.int64(df_train['SHEDID'] == df_train['RTSHEDID'])

        df_train.loc[:, 'lease_monthint'] = df_train['lease_dt'].dt.year*100+df_train['lease_dt'].dt.month
        df_train.loc[:, 'return_monthint'] = df_train['return_dt'].dt.year * 100 + df_train['return_dt'].dt.month

        df_train.loc[:, 'lease_dayint'] = df_train['lease_dt'].dt.year * 100*100 + df_train['lease_dt'].dt.month*100+ df_train['lease_dt'].dt.day
        df_train.loc[:, 'return_dayint'] = df_train['return_dt'].dt.year * 100*100 + df_train['return_dt'].dt.month*100+ df_train['return_dt'].dt.day

        df_train['OPNAME'] = df_train['OPNAME'].map(OPNAME_TO_CODE)

        #df_train = df_train[(df_train.same_shedid == 0)&(df_train.USETIME>1)]

        df_train.to_pickle(file_name)
        return df_train
    else:
        df_shop_info = pd.read_pickle(file_name)
        return df_shop_info

# ----------------------------------------------------------------------------------------------------------------------#
def get_shed_position():
    df_position = pd.read_csv('../data/input/shed_position.csv')
    #df_position.loc[:,'longitude'] = df_position['longitude']*111/15
    #df_position.loc[:, 'latitude'] = df_position['latitude'] * 111/15

    shedid_list = [1,5,6,7,8,9,10,12,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,50,51,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,73,74,75,76,77,78,80,81,83,85,86,88,92,93,95,96,97,98,99,100,101,102,103,105,106,107,108,109,110,111,115,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,138,140,141,142,143,144,145,147,149,150,151,152,153,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,194,195,196,197,198,199,200,201,202,203,209,210,211,213,216,217,218,219,220,221,222,223,224,225,229,231,236,237,239,240,241,242,246,247,248,249,250,252,255,256,260,261,262,263,265,266,268,269,270,271,272,274,275,277,281,285,287,288,290,293,295,297,298,299,300,301,302,303,305,306,308,309,310,311,313,314,315,318,319,320,321,322,323,324,326,329,332,333,334,337,339,342,343,353,354,355,357,359,361,366,367,368,370,378,382,387,395,396,397,398,399,417]

    #df_position.loc[:, 'is_null_position'] = 0
    #df_position.loc[df_position.SHEDID.isin(shedid_list),'is_null_position'] = 1

    df_position = df_position[df_position.SHEDID.isin(shedid_list)]

    return df_position


# ----------------------------------------------------------------------------------------------------------------------#
weather_map_rain = {
    '晴':1,
    '多云':2,
    '阴':3,
    '小雨':4,
    '雷阵雨':5,
    '雷雨':6,
    '中雨':7,
    '大雨':8,
    '暴雨':9
}
def get_weather():
    file_name = '../data/middle/coded_weather.csv'
    if not os.path.exists(file_name):
        df_tianqi = pd.read_csv('../data/input/yancheng.csv', header=None, sep='\t', encoding='UTF-8')
        df_tianqi.columns = ['dayint', 'max_teamp', 'min_teamp', 'weather', 'wind_direction', 'wind_speed']

        df_tianqi.loc[:, 'dayint'] = np.int64(
            df_tianqi['dayint'].apply(lambda x: np.int64(datetime.strptime(x, '%Y-%m-%d').strftime('%Y%m%d'))))

        df_tianqi.loc[:, 'max_teamp'] = np.int64(df_tianqi['max_teamp'])
        df_tianqi.loc[:, 'min_teamp'] = np.int64(df_tianqi['min_teamp'])

        df_tianqi.loc[:, 'weather1'] = df_tianqi['weather'].apply(lambda x: x if '~' not in x else x.split('~')[0])
        df_tianqi.loc[:, 'weather'] = df_tianqi['weather'].apply(lambda x: x if '~' not in x else x.split('~')[1])

        df_tianqi.loc[:, 'wind_speed1'] = df_tianqi['wind_speed'].apply(lambda x: x if '~' not in x else x.split('~')[0])
        df_tianqi.loc[:, 'wind_speed'] = df_tianqi['wind_speed'].apply(lambda x: x if '~' not in x else x.split('~')[1])

        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_tianqi['weather1'].append(df_tianqi['weather']).values))
        df_tianqi['weather1'] = lbl.transform(list(df_tianqi['weather1'].values))
        df_tianqi['weather'] = lbl.transform(list(df_tianqi['weather'].values))

        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_tianqi['wind_speed1'].append(df_tianqi['wind_speed']).values))
        df_tianqi['wind_speed1'] = lbl.transform(list(df_tianqi['wind_speed1'].values))
        df_tianqi['wind_speed'] = lbl.transform(list(df_tianqi['wind_speed'].values))

        #print(df_tianqi['weather'].unique())
        #print(df_tianqi['weather1'].unique())
        # ['小雨' '阴' '多云' '晴' '雷雨' '中雨' '大雨' '暴雨']
        # ['多云' '中雨' '晴' '阴' '小雨' '雷雨' '大雨' '暴雨' '雷阵雨']

        #df_tianqi.loc[:,'rain1'] = np.int64(df_tianqi['weather1'].str.contains('雨'))
        #df_tianqi.loc[:, 'rain'] = np.int64(df_tianqi['weather'].str.contains('雨'))

        #df_tianqi.loc[:, 'rain1'] = df_tianqi['weather1'].map(weather_map_rain)
        #df_tianqi.loc[:, 'rain'] = df_tianqi['weather'].map(weather_map_rain)
        df_tianqi = pd.get_dummies(df_tianqi,columns=['weather','weather1','wind_speed','wind_speed1'])



        df_tianqi.loc[:, 'wind_direction1'] = df_tianqi['wind_direction'].apply(
            lambda x: x if '~' not in x else x.split('~')[0])
        df_tianqi.loc[:, 'wind_direction'] = df_tianqi['wind_direction'].apply(
            lambda x: x if '~' not in x else x.split('~')[1])

        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_tianqi['wind_direction1'].append(df_tianqi['wind_direction']).values))
        df_tianqi['wind_direction1'] = lbl.transform(list(df_tianqi['wind_direction1'].values))
        df_tianqi['wind_direction'] = lbl.transform(list(df_tianqi['wind_direction'].values))



        df_tianqi.to_csv(file_name,index=False)
        return df_tianqi
    else:
        df_tianqi =pd.read_csv(file_name)
        return df_tianqi

# ----------------------------------------------------------------------------------------------------------------------#
def get_ext_time():

    df_test_shedid = get_test_data()
    shedid = df_test_shedid.SHEDID.unique()
    df_shedid = DataFrame({'SHEDID':shedid,'merge_id':1})
    #df_shedid.drop_duplicates(inplace=True)

    df_ext_time = DataFrame({'datetime':pd.date_range('2015-01-01 0:00:00','2015-10-31 23:00:00',freq='D'),'merge_id':1})
    df_ext_time.loc[:,'ext_datetime'] = df_ext_time['datetime'].apply(lambda x:('{}/{}/{}'.format(x.year,x.month,x.day)))
    df_ext_time.loc[:, 'dayint'] = df_ext_time['datetime'].dt.year*100*100+df_ext_time['datetime'].dt.month*100+df_ext_time['datetime'].dt.day

    del df_ext_time['datetime']

    df_ext_time = df_ext_time.merge(df_shedid,on='merge_id',how='outer')
    df_ext_time = df_ext_time[['SHEDID','ext_datetime','dayint']].drop_duplicates()

    #print(df_ext_time.dtypes)
    #df_cnt = df_ext_time.groupby('SHEDID')['dayint'].count().reset_index()
    #df_cnt.columns =['SHEDID','day_cnt']
    #df_cnt.sort_values(by='day_cnt',ascending=False,inplace=True)
    #print(df_cnt)

    return df_ext_time

# ----------------------------------------------------------------------------------------------------------------------#
def get_holiday_features():
    df_ext_time = get_ext_time()

    df_ext_time = df_ext_time[['dayint']].drop_duplicates()

    # 星期几
    df_ext_time.loc[:, 'dt'] = df_ext_time['dayint'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    df_ext_time.loc[:, 'weekday'] = df_ext_time['dt'].dt.dayofweek + 1
    df_ext_time.loc[:, 'weekend'] = np.int64(df_ext_time['weekday'] >= 6)

    df_ext_time = pd.get_dummies(df_ext_time, columns=['weekday'], prefix='weekday')

    # get_dummies会去掉weekday，所以要重新加上
    df_ext_time.loc[:, 'weekday'] = df_ext_time['dt'].dt.dayofweek + 1

    holiday_day_set = [20150101, 20150102, 20150103, 20150110, 20150111, 20150117, 20150118, 20150124, 20150125,
                       20150131, 20150201,
                       20150207, 20150208, 20150218, 20150219, 20150220, 20150221, 20150222, 20150223, 20150224,
                       20150301, 20150307, 20150308, 20150314, 20150315, 20150321, 20150322, 20150328, 20150329,
                       20150404, 20150405, 20150406, 20150411, 20150412, 20150418, 20150419, 20150425, 20150426,
                       20150501, 20150502, 20150503, 20150509, 20150510, 20150516, 20150517, 20150523, 20150524,
                       20150530, 20150531, 20150606, 20150607, 20150613, 20150614, 20150620, 20150621, 20150622,
                       20150627, 20150628, 20150704, 20150705, 20150711, 20150712, 20150718, 20150719, 20150725,
                       20150726, 20150801, 20150802, 20150808, 20150809, 20150815, 20150816, 20150822, 20150823,
                       20150829, 20150830, 20150903, 20150904, 20150905, 20150906, 20150912, 20150913, 20150919,
                       20150920, 20150926, 20150927, 20151001, 20151002, 20151003, 20151004, 20151005, 20151006,
                       20151007, 20151011, 20151017, 20151018, 20151024, 20151025]

    df_ext_time.loc[:, 'holiday'] = 0
    df_ext_time.loc[df_ext_time.dayint.isin(holiday_day_set), 'holiday'] = 1

    holiday_set = [20150101, 20150102, 20150103, 20150218, 20150219, 20150220, 20150221, 20150222, 20150223, 20150224,
                   20150404, 20150405, 20150406,
                   20150501, 20150502, 20150503, 20150620, 20150621, 20150622, 20150927, 20151001, 20151002, 20151003,
                   20151004, 20151005, 20151006, 20151007]

    df_ext_time.loc[df_ext_time.dayint.isin(holiday_set), 'holiday'] = 2

    work_workend_set = [20150104, 20150215, 20150228, 20151010]

    df_ext_time.loc[:, 'workday'] = 1
    df_ext_time.loc[df_ext_time.dayint.isin(holiday_set), 'workday'] = 0
    df_ext_time.loc[df_ext_time.weekday >= 6, 'workday'] = 0
    df_ext_time.loc[df_ext_time.dayint.isin(work_workend_set), 'workday'] = 1

    df_ext_time.loc[:, 'holiday1'] = 0
    df_ext_time.loc[df_ext_time.dayint.isin(holiday_set), 'holiday1'] = 1

    feature = [x for x in df_ext_time.columns if
               x not in ['SHEDID', 'ext_datetime', 'dt']]

    df_ext_time = df_ext_time[feature].drop_duplicates()

    return df_ext_time

# ----------------------------------------------------------------------------------------------------------------------#
def get_daily_train():
    file_name = '../data/middle/daily_train.csv'
    if not os.path.exists(file_name):
        df_train = get_train_data()
        df_ext_dt = get_ext_time()
        df_ext_dt.columns = ['SHEDID','time','dayint']

        df_lease_label = df_train.groupby(['SHEDID', 'lease_dayint'])['LOCATIVEID'].count().reset_index()
        df_lease_label.columns = ['SHEDID', 'dayint', 'lease_cnt']
        df_return_label = df_train.groupby(['RTSHEDID', 'return_dayint'])['LOCATIVEID'].count().reset_index()
        df_return_label.columns = ['SHEDID', 'dayint', 'return_cnt']

        df_lease_label1 = df_ext_dt.merge(df_lease_label, on=['SHEDID', 'dayint'], how='left')
        df_lease_label = df_lease_label.merge(df_return_label, on=['SHEDID', 'dayint'], how='left')
        df_lease_label.fillna(0, inplace=True)

        df_lease_label.loc[:,'lease_cnt'] = np.int64(df_lease_label['lease_cnt'])
        df_lease_label.loc[:, 'return_cnt'] = np.int64(df_lease_label['return_cnt'])

        df_lease_label.to_csv(file_name,index=False)
        return df_lease_label
    else:
        df_lease_label = pd.read_csv(file_name)
        return df_lease_label

# ----------------------------------------------------------------------------------------------------------------------#
def get_topn_rental_shedid(topN):
    df_train = get_daily_train()
    df_train.loc[:, 'rental_cnt'] = df_train['lease_cnt'] + df_train['return_cnt']
    df_stat = df_train.groupby('SHEDID')['rental_cnt'].sum().reset_index()
    df_stat.columns = ['SHEDID', 'total_cnt']

    df_stat.sort_values(by=['total_cnt'], ascending=False, inplace=True)

    return df_stat[:topN]


# ----------------------------------------------------------------------------------------------------------------------#
# 主程序
if __name__ == '__main__':

    #data_exploration()
    #df_daily_label = get_daily_train()
    #print(df_daily_label)

    '''
    df_train = get_train_data()
    df_use_time = df_train.groupby(['USETIME','same_shedid'])['SHEDID'].count().reset_index()
    df_use_time.columns = ['use_time','same_shedid','cnt']
    df_use_time.sort_values(by='cnt',ascending=False,inplace=True)
    df_use_time.to_csv('../data/temp/df_use_time.csv',index=False)
    print(df_use_time)
    '''

    #df_topN = get_topn_rental_shedid(30)
    #print(df_topN)

    df_test = get_test_data()
    df_position = get_shed_position()

    df_test = df_test.merge(df_position,on=['SHEDID'],how='left')
    print(df_test[df_test.longitude.isnull()].SHEDID.unique())
    print(df_test[df_test.longitude.isnull()].SHEDID.nunique())


