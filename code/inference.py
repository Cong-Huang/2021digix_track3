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
import os
import pickle
from utils import reduce_mem, uAUC, ProNE, HyperParam, get_logger, fast_auc

pd.set_option('display.max_columns', None)


valid_pred_lgb1 = pd.read_feather("../data/submit/valid_pred_lgb_1.feather")
valid_pred_lgb2 = pd.read_feather("../data/submit/valid_pred_lgb_2.feather")
valid_pred_nn = pd.read_feather("../data/submit/valid_pred_nn_1.feather")
print(valid_pred_lgb1.shape, valid_pred_lgb2.shape, valid_pred_nn.shape)

test_pred_lgb1 = pd.read_feather("../data/submit/test_pred_lgb_1.feather")
test_pred_lgb2 = pd.read_feather("../data/submit/test_pred_lgb_2.feather")
test_pred_nn = pd.read_feather("../data/submit/test_pred_nn_1.feather")
print(test_pred_lgb1.shape, test_pred_lgb2.shape, test_pred_nn.shape)

valid_pred_lgb = valid_pred_lgb1.copy()
test_pred_lgb = test_pred_lgb1.copy()
for col in ['is_share_pred',
        'watch_label_pred_1', 'watch_label_pred_2', 'watch_label_pred_3',
        'watch_label_pred_4', 'watch_label_pred_5', 'watch_label_pred_6',
        'watch_label_pred_7', 'watch_label_pred_8', 'watch_label_pred_9']:
    valid_pred_lgb[col] = valid_pred_lgb1[col] * 0.3 + valid_pred_lgb2[col] * 0.7
    test_pred_lgb[col] = test_pred_lgb1[col] * 0.3 + test_pred_lgb2[col] * 0.7


def onehot_encode(nums, k):
    res = np.zeros((len(nums), k))
    for i, x in (enumerate(nums)):
        res[i, int(x)] = 1
    res = res.astype(np.int32)
    return res

watch_y_true = onehot_encode(valid_pred_lgb1['watch_label'].values, 10)

def get_all_auc(df):
    auc_list = [roc_auc_score(df['is_share'].values, df['is_share_pred'].values)]
    for i in range(1, 10):
        auc_list.append(fast_auc(watch_y_true[:, i], df['watch_label_pred_{}'.format(i)]))
    return list(np.round(auc_list, 6))

print(get_all_auc(valid_pred_lgb))
print(get_all_auc(valid_pred_nn))


valid_pred_ronghe = valid_pred_lgb.copy()
test_pred_ronghe = test_pred_lgb.copy()

for col in ['is_share_pred',
        'watch_label_pred_1', 'watch_label_pred_2', 'watch_label_pred_3',
        'watch_label_pred_4', 'watch_label_pred_5', 'watch_label_pred_6',
        'watch_label_pred_7', 'watch_label_pred_8', 'watch_label_pred_9']:
    if col == 'is_share_pred':
        valid_pred_ronghe[col] = valid_pred_nn[col] * 0.15 + valid_pred_lgb[col] * 0.85
        test_pred_ronghe[col] = test_pred_nn[col] * 0.15 + test_pred_lgb[col] * 0.85
    else:
        valid_pred_ronghe[col] = valid_pred_nn[col] * 0.05 + valid_pred_lgb[col] * 0.95
        test_pred_ronghe[col] = test_pred_nn[col] * 0.05 + test_pred_lgb[col] * 0.95
    
print(get_all_auc(valid_pred_ronghe))


## 线下后处理和评测
def get_deal_y_pred(valid_14):
    nums = valid_14.shape[0]    # 数据集大小
    ## 每类的取值个数 [9,8,7,6,5,4,3,2,1]
    class_nums = np.array([int(nums*0.035)]*9) * np.array([3.5, 1.4, 1.3, 1.2, 1.1, 1.5, 1.0, 1.0, 1.5])
    ## 需要返回的结果
    y_pred_res = np.zeros(nums)
    
    cls_2_998fws = {i: np.percentile(valid_14['watch_label_pred_{}'.format(i)], 99.9) 
                   for i in range(1, 10)}   # 每个类对应的998分位数  概率更大
    
    cls_2_996fws = {i: np.percentile(valid_14['watch_label_pred_{}'.format(i)], 99.7) 
                   for i in range(1, 10)}   # 每个类对应的996分位数  概率更大
    
    cls_2_99fws = {i: np.percentile(valid_14['watch_label_pred_{}'.format(i)], 99) 
                   for i in range(1, 10)}   # 每个类对应的99分位数  概率更大
    
    cls_2_97fws = {i: np.percentile(valid_14['watch_label_pred_{}'.format(i)], 97) 
                   for i in range(1, 10)}   # 每个类对应的97分位数  概率更小
    
    cls_2_95fws = {i: np.percentile(valid_14['watch_label_pred_{}'.format(i)], 95) 
                   for i in range(1, 10)}   # 每个类对应的95分位数  概率更小
    
    cls_2_90fws = {i: np.percentile(valid_14['watch_label_pred_{}'.format(i)], 90) 
                   for i in range(1, 10)}   # 每个类对应的90分位数  概率更小
    
    idx2cls_prob = {}   ## 已经访问过的记录： index -> (cls, prob)
    
    for i, n in enumerate(class_nums):
        probs = valid_14['watch_label_pred_{}'.format(9-i)]   # 概率，最先是第9类
        idxes = np.argsort(-1 * probs).values
        
        idx = 0   # 开始遍历的索引的位置，需要赋值到答案中
        cnt = 0   # 总共遍历得到的个数
        while cnt < n:
            if idxes[idx] not in idx2cls_prob:
                y_pred_res[idxes[idx]] = 9 - i
                cnt += 1
                idx2cls_prob[idxes[idx]] = (9-i, probs[idxes[idx]])
            else:
                flag = False
                prob = idx2cls_prob[idxes[idx]][1]
                if cnt <= (nums * 0.002) and i <= 5 and  prob<=cls_2_998fws[idx2cls_prob[idxes[idx]][0]]:
                    flag = True
                elif cnt <= (nums * 0.005) and i <= 5 and  prob<=cls_2_996fws[idx2cls_prob[idxes[idx]][0]]:
                    flag = True
                elif cnt <= (nums * 0.01) and i <= 5 and prob <= cls_2_99fws[idx2cls_prob[idxes[idx]][0]]:
                    flag = True
                elif cnt <= (nums * 0.02) and i <= 5 and prob <= cls_2_97fws[idx2cls_prob[idxes[idx]][0]]:
                    flag = True
                elif cnt <= (nums * 0.001) and i > 5 and prob <= cls_2_97fws[idx2cls_prob[idxes[idx]][0]]:
                    flag = True
                elif cnt <= (nums * 0.003) and i > 5 and prob <= cls_2_95fws[idx2cls_prob[idxes[idx]][0]]:
                    flag = True
                elif cnt <= (nums * 0.005) and i > 5 and prob <= cls_2_90fws[idx2cls_prob[idxes[idx]][0]]:
                    flag = True
                if flag:
                    y_pred_res[idxes[idx]] = 9 - i
                    cnt += 1
                    idx2cls_prob[idxes[idx]] = (9-i, probs[idxes[idx]])
            idx += 1
        print('cls:', 9-i, idx, cnt)
    print(pd.Series(y_pred_res).value_counts())
    return y_pred_res



def calc_weighted_auc(valid_14):
    ## 计算 watch_label的AUC
    valid_14['watch_label_pred'] = get_deal_y_pred(valid_14)
    watch_y_pred = onehot_encode(valid_14['watch_label_pred'].values, 10)
    auc_list = []
    for i in range(1, 10):
        score = fast_auc(watch_y_true[:, i], watch_y_pred[:, i])
        auc_list.append(score)
    
    y2_auc = sum(np.array(auc_list) * np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]))
    y1_auc = roc_auc_score(valid_14['is_share'], valid_14['is_share_pred'])
    weighted_auc = y1_auc * 0.3 + y2_auc * 0.7
    return weighted_auc, list(np.round([y1_auc] + auc_list, 4))

wauc, auc_list = calc_weighted_auc(valid_pred_ronghe)
print("*** 验证集上的weighted auc {}, auc_list {}".format(wauc, auc_list))

## 得到最后的提交文件submission.csv
print("得到最后的提交文件，即data/submission.csv")
test_pred_ronghe['watch_label'] = get_deal_y_pred(test_pred_ronghe)
submit_final = test_pred_ronghe[['userid', 'feedid', 'watch_label', 'is_share_pred']]
submit_final.columns = ['user_id', 'video_id', 'watch_label', 'is_share']
submit_final['is_share'] = np.round(submit_final['is_share'], 8)
submit_final['watch_label'] = submit_final['watch_label'].astype(int)

submit_final.to_csv("../data/submission.csv", index=None)




