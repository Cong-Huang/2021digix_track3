# coding=UTF-8

import pandas as pd 
import numpy as np
import os
import time 
import gc
from tqdm import tqdm
tqdm.pandas()
from utils import reduce_mem, uAUC, ProNE, HyperParam, get_logger
from sklearn.metrics import *
import warnings

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


start_time = time.time()
## 读取embedding特征
# for lgb
fid_w2v_emb = pd.read_pickle("../data/features/lgb_emb/fid_w2v_emb_lgb.pkl")
fid_tfidf_svd_emb = pd.read_pickle("../data/features/lgb_emb/fid_tfidf_svd_emb_lgb.pkl")
fid_prone_emb = pd.read_pickle("../data/features/lgb_emb/feed_prone_emb_lgb.pkl")
fid_tag_emb = pd.read_pickle("../data/features/lgb_emb/fid_tag_svd_emb_lgb.pkl")
fid_desc_emb = pd.read_pickle("../data/features/lgb_emb/fid_desc_svd_emb_lgb.pkl")

uid_tfidf_svd_emb = pd.read_pickle("../data/features/lgb_emb/uid_tfidf_svd_emb_lgb.pkl")
uid_prone_emb = pd.read_pickle("../data/features/lgb_emb/user_prone_emb_lgb.pkl")
uid_sif_hist_emb = pd.read_pickle("../data/features/lgb_emb/uid_sif_hist_emb_lgb.pkl")
print(fid_w2v_emb.shape, fid_tfidf_svd_emb.shape, fid_prone_emb.shape, fid_tag_emb.shape, fid_desc_emb.shape)
print(uid_tfidf_svd_emb.shape, uid_prone_emb.shape, uid_sif_hist_emb.shape)

fid_all_emb = fid_w2v_emb.merge(fid_tfidf_svd_emb, how='left', on=['feedid'])
fid_all_emb = fid_all_emb.merge(fid_prone_emb, how='left', on=['feedid'])
fid_all_emb = fid_all_emb.merge(fid_tag_emb, how='left', on=['feedid'])
fid_all_emb = fid_all_emb.merge(fid_desc_emb, how='left', on=['feedid'])
fid_all_emb.fillna(0.0, inplace=True)
print(fid_all_emb.shape)

uid_all_emb = uid_tfidf_svd_emb.merge(uid_prone_emb, how='left', on=['userid'])
uid_all_emb = uid_all_emb.merge(uid_sif_hist_emb, how='left', on=['userid'])
print(uid_all_emb.shape)

uid_all_emb = reduce_mem(uid_all_emb)
fid_all_emb = reduce_mem(fid_all_emb)

uid_all_emb.to_feather("../data/features/uid_all_emb_lgb.feather")
fid_all_emb.to_feather("../data/features/fid_all_emb_lgb.feather")


print("整理数据，变成模型所需要的形式")
df1 = pd.read_feather("../data/features/df_stat_v1.feather")
df2 = pd.read_feather("../data/features/df_stat_v2.feather")
df3 = pd.read_feather("../data/features/df_stat_v3.feather")
print(df1.shape, df2.shape, df3.shape)

del_cols = ['is_watch', 'is_collect', 'is_comment', 'watch_playseconds',
            'watch_label_cls_1', 'watch_label_cls_2', 'watch_label_cls_3', 
            'watch_label_cls_4', 'watch_label_cls_5', 'watch_label_cls_6',
            'watch_label_cls_7', 'watch_label_cls_8', 'watch_label_cls_9']
df1.drop(columns=del_cols, inplace=True)

## 拼接特征df2
df = pd.concat([df1, df2[df2.columns[31:]]], axis=1)
del df1, df2
gc.collect()

## 拼接特征df3
df = pd.concat([df, df3[df3.columns[21:]]], axis=1)
print(df.shape)
del df3
gc.collect()


## 去除第一天的样本特征
df = df[df['date_'] != 1].reset_index(drop=True)
print(df.shape)
df.to_feather("../data/features/df_stat_final.feather")


## 整理得到所需要训练的数据
def get_train_sampling(df, frac_rate=0.15, seed=2021):
    train_tmp = df[df['date_']<=13]
    del df
    gc.collect()

    y_list = ['is_share', 'watch_label']
    train_tmp['y_sum'] = train_tmp[y_list].sum(axis=1)
    ## 正样本不采样
    train_pos = train_tmp[train_tmp['y_sum'] >= 1]
    ## 负样本采样x%
    train_neg = train_tmp[train_tmp['y_sum'] == 0].sample(frac=frac_rate, random_state=seed)
    del train_tmp
    gc.collect()

    print("正负样本个数：", train_pos.shape, train_neg.shape)
    train = pd.concat([train_pos, train_neg], ignore_index=True)
    train = train.sample(frac=1.0, random_state=2021).reset_index(drop=True)
    
    del train_pos, train_neg, train['y_sum']
    gc.collect()
    
    print("train shape: {}".format(train.shape))
    return train


print("开始对数据进行采样...")
df = pd.read_feather("../data/features/df_stat_final.feather")
valid_14 = df[df['date_'] == 14].reset_index(drop=True)
test = df[df['date_'] == 15].reset_index(drop=True)
print(valid_14.shape, test.shape)
valid_14.to_feather("../data/model_data/valid_14.feather")
test.to_feather("../data/model_data/test.feather")
del valid_14, test
gc.collect()

train_lgb1 = get_train_sampling(df, frac_rate=0.15, seed=2021)
train_lgb1.to_feather("../data/model_data/train_lgb1.feather")
del train_lgb1
gc.collect()

train_lgb2 = get_train_sampling(df, frac_rate=0.15, seed=6666)
train_lgb2.to_feather("../data/model_data/train_lgb2.feather")
del train_lgb2
gc.collect()

print("time costed {}(s)".format(round(time.time() - start_time, 6)))






