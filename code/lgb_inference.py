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
import time
import pickle
from utils import reduce_mem, uAUC, ProNE, HyperParam, get_logger, fast_auc

pd.set_option('display.max_columns', None)

logger = get_logger("../data/log/lgb_inference.txt")


logger.info("读取数据...")
valid_14 = pd.read_feather("../data/model_data/valid_14.feather")
test = pd.read_feather("../data/model_data/test.feather")

# For debug
#valid_14 = valid_14.sample(n=20000).reset_index(drop=True)
#test = test.sample(n=10000).reset_index(drop=True)

logger.info("读取embedding数据...")
uid_all_emb = pd.read_feather("../data/features/uid_all_emb_lgb.feather")
fid_all_emb = pd.read_feather("../data/features/fid_all_emb_lgb.feather")
logger.info("uid_all_emb.shape {}, fid_all_emb.shape {}".format(uid_all_emb.shape, fid_all_emb.shape))

def merge_emb_df(df):
    df = df.merge(uid_all_emb, how='left', on=['userid'])
    df = df.merge(fid_all_emb, how='left', on=['feedid'])
    return df
valid_14 = merge_emb_df(valid_14)
test = merge_emb_df(test)

logger.info('valid.shape {}, test.shape {}'.format(valid_14.shape, test.shape))
del uid_all_emb, fid_all_emb
gc.collect()

## lgb训练模型所需要的特征列
y_list = ['is_watch', 'is_share', 'is_collect', 'is_comment', 'watch_label']
cols = [f for f in test.columns if (f not in ['date_'] + y_list)]
logger.info("特征总数：{}".format(len(cols)))



logger.info("开始预测is_share......")
start_time = time.time()
clf_share = pickle.load(open("../data/save_model/lgb_model_is_share_1.pkl", 'rb'))
valid_14['is_share_pred'] = clf_share.predict_proba(valid_14[cols].values)[:, 1]
test['is_share_pred'] = clf_share.predict_proba(test[cols].values)[:, 1]
logger.info("预测耗时: {}(s)".format(round(time.time() - start_time, 6)))

logger.info("开始预测watch_label......")
start_time = time.time()
clf_watch = pickle.load(open("../data/save_model/lgb_model_watch_label_1.pkl", 'rb'))
valid_14_watch_pred = clf_watch.predict_proba(valid_14[cols].values.astype(np.float32))
test_watch_pred = clf_watch.predict_proba(test[cols].values.astype(np.float32))
logger.info("预测耗时: {}(s)".format(round(time.time() - start_time, 6)))



logger.info("保存数据")
valid_pred_lgb = valid_14[['userid', 'feedid', 'watch_label', 'is_share', 'is_share_pred']]
tmp = pd.DataFrame(valid_14_watch_pred[:, 1:], columns=['watch_label_pred_1', 'watch_label_pred_2', 'watch_label_pred_3', 
                                                        'watch_label_pred_4', 'watch_label_pred_5', 'watch_label_pred_6',
                                                        'watch_label_pred_7', 'watch_label_pred_8', 'watch_label_pred_9'])
valid_pred_lgb = pd.concat([valid_pred_lgb, tmp], axis=1)

test_pred_lgb = test[['userid', 'feedid', 'is_share_pred']]
tmp = pd.DataFrame(test_watch_pred[:, 1:], columns=['watch_label_pred_1', 'watch_label_pred_2', 'watch_label_pred_3', 
                                                    'watch_label_pred_4', 'watch_label_pred_5', 'watch_label_pred_6',
                                                    'watch_label_pred_7', 'watch_label_pred_8', 'watch_label_pred_9'])
test_pred_lgb = pd.concat([test_pred_lgb, tmp], axis=1)
valid_pred_lgb.to_feather("../data/submit/valid_pred_lgb_1.feather")
test_pred_lgb.to_feather("../data/submit/test_pred_lgb_1.feather")





logger.info("开始预测is_share......")
start_time = time.time()
clf_share = pickle.load(open("../data/save_model/lgb_model_is_share_2.pkl", 'rb'))
valid_14['is_share_pred'] = clf_share.predict_proba(valid_14[cols].values)[:, 1]
test['is_share_pred'] = clf_share.predict_proba(test[cols].values)[:, 1]
logger.info("预测耗时: {}(s)".format(round(time.time() - start_time, 6)))

logger.info("开始预测watch_label......")
start_time = time.time()
clf_watch = pickle.load(open("../data/save_model/lgb_model_watch_label_2.pkl", 'rb'))
valid_14_watch_pred = clf_watch.predict_proba(valid_14[cols].values.astype(np.float32))
test_watch_pred = clf_watch.predict_proba(test[cols].values.astype(np.float32))
logger.info("预测耗时: {}(s)".format(round(time.time() - start_time, 6)))



logger.info("保存数据")
valid_pred_lgb = valid_14[['userid', 'feedid', 'watch_label', 'is_share', 'is_share_pred']]
tmp = pd.DataFrame(valid_14_watch_pred[:, 1:], columns=['watch_label_pred_1', 'watch_label_pred_2', 'watch_label_pred_3', 
                                                        'watch_label_pred_4', 'watch_label_pred_5', 'watch_label_pred_6',
                                                        'watch_label_pred_7', 'watch_label_pred_8', 'watch_label_pred_9'])
valid_pred_lgb = pd.concat([valid_pred_lgb, tmp], axis=1)

test_pred_lgb = test[['userid', 'feedid', 'is_share_pred']]
tmp = pd.DataFrame(test_watch_pred[:, 1:], columns=['watch_label_pred_1', 'watch_label_pred_2', 'watch_label_pred_3', 
                                                    'watch_label_pred_4', 'watch_label_pred_5', 'watch_label_pred_6',
                                                    'watch_label_pred_7', 'watch_label_pred_8', 'watch_label_pred_9'])
test_pred_lgb = pd.concat([test_pred_lgb, tmp], axis=1)
valid_pred_lgb.to_feather("../data/submit/valid_pred_lgb_2.feather")
test_pred_lgb.to_feather("../data/submit/test_pred_lgb_2.feather")



