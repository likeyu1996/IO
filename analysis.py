#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 李珂宇
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

summary_df = pd.read_csv('./Result/summary_df.csv', index_col=0)
# print(summary_df)

all_date = sorted(np.array(list(set(summary_df['TradingDate'].to_numpy()))), reverse=True)
all_option = sorted(np.array(list(set(summary_df['Symbol'].to_numpy()))))
print(all_date)
print(all_option)

for i in all_date:
    df_cache_day = summary_df[summary_df['TradingDate'] == i]
    plt.figure(figsize=(16, 9))
    plt.title('Imp_Vol of {0}'.format(i))
    sns.lineplot(data=df_cache_day, x='StrikePrice', y='ImpVolatility',
                 hue='CallOrPut', hue_order=['P', 'C'], sort=True, estimator=None)
    plt.savefig('./Result/{0}/{0}{1}'.format('imp_vol', i))
    plt.close('all')
'''

for i in all_date:
    df_cache_day = summary_df[summary_df['TradingDate'] == i]
    plt.figure(figsize=(16, 9))
    plt.title('Diff_in_Days of {0}'.format(i))
    sns.lineplot(data=df_cache_day, x='Value_in_Value', y='P-P_BS_easy',
                 hue='CallOrPut', hue_order=['P', 'C'], sort=True, estimator=None)
    plt.savefig('./Result/{0}/{0}{1}'.format('diff_in_days', i))
    plt.close('all')
'''
'''
for i in all_option:
    df_cache_option = summary_df[summary_df['Symbol'] == i]
    plt.figure(figsize=(16, 9))
    plt.title('Diff_over_Days of {0}'.format(i))
    sns.lineplot(data=df_cache_option, x='Value_in_Value', y='P-P_BS_easy',
                 hue='CallOrPut', hue_order=['P', 'C'], sort=True, estimator=None)
    plt.savefig('./Result/{0}/{0}{1}'.format('diff_over_days', i))
    plt.close('all')
'''
