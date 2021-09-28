# coding=UTF-8

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold 
import gc
import random
import time
import pickle
from utils import reduce_mem, uAUC, ProNE, HyperParam, get_logger, fast_auc

pd.set_option('display.max_columns', None)


logger = get_logger("../data/log/lgb_train_log.txt")

logger.info("读取数据...")
valid_14 = pd.read_feather("../data/model_data/valid_14.feather")
#valid_14 = valid_14.sample(n=10000).reset_index(drop=True)   # debug
logger.info('valid_14.shape {}'.format(valid_14.shape))


logger.info("读取embedding数据...")
uid_all_emb = pd.read_feather("../data/features/uid_all_emb_lgb.feather")
fid_all_emb = pd.read_feather("../data/features/fid_all_emb_lgb.feather")
logger.info("uid_all_emb.shape {}, fid_all_emb.shape {}".format(uid_all_emb.shape, fid_all_emb.shape))


def merge_emb_df(df):
    df = df.merge(uid_all_emb, how='left', on=['userid'])
    df = df.merge(fid_all_emb, how='left', on=['feedid'])
    return df


valid_14 = merge_emb_df(valid_14)
logger.info('valid_14.shape {}'.format(valid_14.shape))


cate_cols = ['userid', 'feedid', 'age', 'gender', 'country', 'province',
             'city', 'city_level', 'device_name']
y_list = ['is_watch', 'is_share', 'is_collect', 'is_comment', 'watch_label']


## lgb训练模型所需要的特征列
cols = [f for f in valid_14.columns if (f not in ['date_'] + y_list)]
logger.info("特征总数：{}".format(len(cols)))



def lgb_train_is_share(train, valid_14):    
    clf_share = LGBMClassifier(
                learning_rate=0.01,
                n_estimators=2000,
                num_leaves=127,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=2021,
                min_child_samples=50,
                metric='None',
                importance_type='gain',
                n_jobs=32)
    
    clf_share.fit(train[cols].values.astype(np.float32), 
                train['is_share'].values,
                eval_set=[(train[cols].values.astype(np.float32),  train['is_share'].values), 
                          (valid_14[cols].values.astype(np.float32),  valid_14['is_share'].values)],
                eval_names=['train', 'valid'],
                eval_metric='auc',
                early_stopping_rounds=50,
                verbose=50)
    return clf_share 


def lgb_train_watch_label(train, valid_14):
    clf_watch = LGBMClassifier(
            learning_rate=0.06,
            objective='multiclass',
            n_estimators=1500,
            num_leaves=127,
            subsample=0.80,
            colsample_bytree=0.85,
            random_state=2021,
            metric='None',
            n_jobs=32)
        
    clf_watch.fit(train[cols].values.astype(np.float32),
                train['watch_label'].values.astype(np.int32),
                eval_set=[(valid_14[cols].values.astype(np.float32),
                           valid_14['watch_label'].values.astype(np.int32))],
                eval_metric='logloss',
                early_stopping_rounds=20,
                verbose=50)
    return clf_watch



logger.info("开始训练is_share......")
start_time = time.time()
train = pd.read_feather("../data/model_data/train_lgb1.feather")
#train = train.sample(n=20000).reset_index(drop=True)   # For debug

train = merge_emb_df(train)
logger.info("train shape: {}".format(train.shape))
logger.info("is_share 标签分布: \n {}".format(train['is_share'].value_counts()))
clf_share = lgb_train_is_share(train, valid_14)
pickle.dump(clf_share, open("../data/save_model/lgb_model_is_share_1.pkl", 'wb'))
logger.info("训练耗时: {}(s)".format(round(time.time() - start_time, 6)))

logger.info("开始训练watch_label......")
start_time = time.time()
logger.info("watch_label 标签分布: \n {}".format(train['watch_label'].value_counts()))
clf_watch = lgb_train_watch_label(train, valid_14)
pickle.dump(clf_watch, open("../data/save_model/lgb_model_watch_label_1.pkl", 'wb'))
logger.info("训练耗时: {}(s)".format(round(time.time() - start_time, 6)))

del train
gc.collect()


logger.info("开始训练is_share......")
start_time = time.time()
train = pd.read_feather("../data/model_data/train_lgb2.feather")
#train = train.sample(n=20000).reset_index(drop=True)   # For debug

train = merge_emb_df(train)
logger.info("train shape: {}".format(train.shape))
logger.info("is_share 标签分布: \n {}".format(train['is_share'].value_counts()))
clf_share = lgb_train_is_share(train, valid_14)
pickle.dump(clf_share, open("../data/save_model/lgb_model_is_share_2.pkl", 'wb'))
logger.info("训练耗时: {}(s)".format(round(time.time() - start_time, 6)))


logger.info("开始训练watch_label......")
start_time = time.time()
logger.info("watch_label 标签分布: \n {}".format(train['watch_label'].value_counts()))
clf_watch = lgb_train_watch_label(train, valid_14)
pickle.dump(clf_watch, open("../data/save_model/lgb_model_watch_label_2.pkl", 'wb'))
logger.info("训练耗时: {}(s)".format(round(time.time() - start_time, 6)))


