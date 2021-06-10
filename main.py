#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 李珂宇
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import sys
import time
import os
import psutil
from chinese_calendar import is_holiday, is_workday
from scipy import stats

# numpy完整print输出
np.set_printoptions(threshold=np.inf)
# pandas完整print输出
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


# TODO 用装饰器实现
def memory_info():
    # 模获得当前进程的pid
    pid = os.getpid()
    # 根据pid找到进程，进而找到占用的内存值
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024
    return memory


def read_clean_db(db_name):
    path_root = r"./Data/"
    # 似乎np.nan和'Nan'效果一样 skiprows是神器 gb18030范围比gbk要大
    db = pd.read_csv(path_root + db_name, encoding='gb18030', na_values=np.nan, skiprows=[1, 2])
    # 股票代码填充到六位
    if db_name == 'IO_PRICINGPARAMETER.CSV':
        list_1 = []
        for i in range(db['UnderlyingSecuritySymbol'].size):
            i_str = str(db.loc[i, 'UnderlyingSecuritySymbol'])
            length = len(i_str)
            if length != 6:
                i_new = '0' * (6 - length) + i_str
                list_1.append(i_new)
                # 直接单个赋值效率极低, 放弃, 尝试整列修改效率极大提升
                # dirty_db.loc[i,'Symbol'] = i_new
            else:
                list_1.append(i_str)
        db['UnderlyingSecuritySymbol'] = list_1
        # 剔除IO以外的数据 波浪线作用的取反 但不是not的意思
        size_0 = db['Symbol'].size
        db.drop(index=db.loc[~ db['Symbol'].str.startswith('IO')].index, inplace=True)
        size_1 = db['Symbol'].size
        print('总共清理了{:}行非IO数据'.format(size_0-size_1))
        # 日期格式化
        db['TradingDate'] = pd.to_datetime(db['TradingDate'])
        db['ExerciseDate'] = pd.to_datetime(db['ExerciseDate'])
        # 按日期由近到远排序
        db.sort_values(by='TradingDate',ascending=False,inplace=True)
        db.reset_index(inplace=True, drop=True)
    elif db_name == 'IDX_Idxtrd.CSV':
        list_1 = []
        for i in range(db['Indexcd'].size):
            i_str = str(db.loc[i, 'Indexcd'])
            length = len(i_str)
            if length != 6:
                i_new = '0' * (6 - length) + i_str
                list_1.append(i_new)
                # 直接单个赋值效率极低, 放弃, 尝试整列修改效率极大提升
                # dirty_db.loc[i,'Symbol'] = i_new
            else:
                list_1.append(i_str)
        db['Indexcd'] = list_1
        # 日期格式化
        db['Idxtrd01'] = pd.to_datetime(db['Idxtrd01'])
        # 按日期由近到远排序
        db.sort_values(by='Idxtrd01',ascending=False,inplace=True)
        db.reset_index(inplace=True, drop=True)
        # 计算简单和对数收益率
        simple_return = np.array(db['Idxtrd05']/db['Idxtrd05'].shift(-1) - 1)
        log_return = np.array(np.log(db['Idxtrd05']/db['Idxtrd05'].shift(-1)))
        new_df = pd.DataFrame([simple_return,log_return],index=['simple_r','log_r']).T
        db = pd.concat([db,new_df],axis=1)
        db.drop(len(db)-1,inplace=True)
    return db


db_pricing = read_clean_db('IO_PRICINGPARAMETER.CSV')
db_index = read_clean_db('IDX_Idxtrd.CSV')
# print(db_pricing.head())
# print(db_index.head())


class EmpiricalIO:
    def __init__(self, db_pricing=db_pricing, db_index=db_index):
        self.db_pricing = db_pricing
        self.db_index = db_index
        self.T = np.array([])
        self.S = np.array([])
        self.X = np.array([])
        self.r = np.array([])
        self.sigma = np.array([])
        self.price = np.array([])

    def his_vol(self, n=120, y=252):
        index_log_return = self.db_index['log_r'].to_numpy()
        # print(index_log_return.size)
        index_his_vol_array = np.array([np.std(index_log_return[i:i+n], ddof=1)*np.sqrt(y)
                                        for i in range(index_log_return.size)
                                        if (i+n < index_log_return.size)])
        # print(index_his_vol_array)
        index_his_vol_cache = np.array(list(index_his_vol_array)+[np.nan for i in range(n)])
        self.db_index['his_vol'] = index_his_vol_cache
        # print(self.db_index)

    def draw_dis(self):
        plt.figure(figsize=(16, 9))
        sns.distplot(self.db_index['log_r'], kde=True, norm_hist=True)
        plt.savefig('./Result/index_log_r')

    def set_parameters(self):
        self.T = self.db_pricing['RemainingTerm'].to_numpy()
        self.S = self.db_pricing['UnderlyingScrtClose'].to_numpy()
        self.X = self.db_pricing['StrikePrice'].to_numpy()
        self.r = self.db_pricing['RisklessRate'].to_numpy() * 0.01
        self.sigma = np.array([self.db_index.loc[self.db_index['Idxtrd01'] == i, 'his_vol'].to_numpy()[0]
                               if not is_holiday(i) else np.nan
                               for i in self.db_pricing['TradingDate']])
        # print(sigma)
        '''
        for i in self.db_pricing['TradingDate']:
            # 发现股指数据库有数据缺失 坑爹啊！甚至多达26条
            # 后查证发现是期权数据库中包含了没有交易的假期
            print(i, is_holiday(i))
            if self.db_index.loc[self.db_index['Idxtrd01'] == i, 'his_vol'].to_numpy().size == 0 \
                    and not is_holiday(i):
                print(i)
            # print(self.db_index.loc[self.db_index['Idxtrd01'] == i, 'his_vol'].to_numpy()[0])
        '''

    def price_BS_easy(self):
        d_1 = (np.log(self.S/self.X)+(self.r+0.5*self.sigma**2)*self.T)/(self.sigma*np.sqrt(self.T))
        d_2 = d_1 - self.sigma*np.sqrt(self.T)
        # print(d_1)
        N_d_1 = stats.norm.cdf(d_1)
        N_d_2 = stats.norm.cdf(d_2)
        price = np.array([self.S[i]*N_d_1[i]-self.X[i]*np.exp(-self.r[i]*self.T[i])*N_d_2[i]
                          if self.db_pricing.loc[i, 'CallOrPut'] == 'C' else
                          self.S[i]*(N_d_1[i]-1)-self.X[i]*np.exp(-self.r[i]*self.T[i])*(N_d_2[i]-1)
                          for i in range(self.db_pricing['CallOrPut'].size)])
        self.price = price

    def summary(self):
        value_in_value = np.array([self.db_pricing.loc[i, 'UnderlyingScrtClose']-self.db_pricing.loc[i, 'StrikePrice']
                                   if self.db_pricing.loc[i, 'CallOrPut'] == 'C' else
                                   self.db_pricing.loc[i, 'StrikePrice']-self.db_pricing.loc[i, 'UnderlyingScrtClose']
                                   for i in range(self.db_pricing['CallOrPut'].size)])
        summary_dict = {
            'TradingDate': self.db_pricing['TradingDate'].to_numpy(),
            'Symbol': self.db_pricing['Symbol'].to_numpy(),
            'CallOrPut': self.db_pricing['CallOrPut'].to_numpy(),
            'RemainingTerm': self.db_pricing['RemainingTerm'].to_numpy(),
            'StrikePrice': self.db_pricing['StrikePrice'].to_numpy(),
            'UnderlyingScrtClose': self.db_pricing['UnderlyingScrtClose'].to_numpy(),
            'ClosePrice': self.db_pricing['ClosePrice'].to_numpy(),
            'TheoreticalPrice': self.db_pricing['TheoreticalPrice'].to_numpy(),
            'BS_easy': self.price,
            'S-X': self.db_pricing['UnderlyingScrtClose'].to_numpy()-self.db_pricing['StrikePrice'].to_numpy(),
            'Value_in_Value': value_in_value,
            'P-P_CSMAR': self.db_pricing['ClosePrice'].to_numpy()-self.db_pricing['TheoreticalPrice'].to_numpy(),
            'P-P_BS_easy': self.db_pricing['ClosePrice'].to_numpy()-self.price,
            'HisVolatility': self.db_pricing['HistoricalVolatility'].to_numpy(),
            'HisVol': self.sigma,
            'ImpVolatility': self.db_pricing['ImpliedVolatility'].to_numpy(),
            'Imp-His': self.db_pricing['ImpliedVolatility'].to_numpy()-self.sigma,
        }
        summary_df = pd.DataFrame(summary_dict)
        summary_df.to_csv('./Result/summary_df.csv')

    def test(self):
        self.his_vol()
        # self.draw_dis()
        self.set_parameters()
        self.price_BS_easy()
        self.summary()


if __name__ == '__main__':
    x = EmpiricalIO()
    x.test()
