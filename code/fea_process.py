#coding=UTF-8

import pandas as pd 
import numpy as np
import os
import time 
import gc
from tqdm import tqdm
tqdm.pandas()
from utils import reduce_mem, uAUC, ProNE, HyperParam, get_logger
import logging
from gensim.models import word2vec
import networkx as nx
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn.metrics import *
from sklearn.cluster import KMeans
import warnings
import jieba,re
import collections

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


## 读取数据集
logger = get_logger("../data/log/fea_process_log.txt")
logger.info("读取数据集")
hist_data_path = "../data/origin/traindata/history_behavior_data/"
hist_df_list = []
for file in tqdm(sorted(os.listdir(hist_data_path))):
    tmp_path = hist_data_path + file
    tmp_path = tmp_path + '/' + os.listdir(tmp_path)[0]
    tmp_df = pd.read_csv(tmp_path, sep='\t', low_memory=False)
    del tmp_df['watch_start_time']
    hist_df_list.append(tmp_df)
    del tmp_df
    gc.collect()

train = pd.concat(hist_df_list, ignore_index=True)
train = reduce_mem(train)
test = pd.read_csv("../data/origin/testdata/test.csv")
logger.info("train.shape {}, test.shape {}".format(train.shape, test.shape))

user_info = pd.read_csv("../data/origin/traindata/user_features_data/user_features_data.csv", sep='\t')
video_info = pd.read_csv("../data/origin/traindata/video_features_data/video_features_data.csv", sep='\t')
logger.info("user_info.shape {}, video_info.shape {}".format(user_info.shape, video_info.shape))


## For debug
#train = train.sample(n=1000000).reset_index(drop=True)
#test = test.sample(n=100000).reset_index(drop=True)

train.to_feather("../data/origin/train.feather")
test.to_feather("../data/origin/test.feather")
user_info.to_feather("../data/origin/user_info.feather")
video_info.to_feather("../data/origin/video_info.feather")


## 处理video_info信息表 (文本侧)
logger.info("处理video_info信息表 (文本侧)")
feed_info = pd.read_feather("../data/origin/video_info.feather")

for col in ['video_name', 'video_tags', 'video_description', 
            'video_director_list', 'video_actor_list', 'video_second_class']:
    feed_info[col] = feed_info[col].fillna("")
    
## 分词操作
stopwords = open("stopwords.txt", encoding='utf-8').readlines()
stopwords = set([x.strip() for x in stopwords])
for col in ['video_name', 'video_description']:
    feed_info[col] = feed_info[col].apply(lambda line: ' '.join([x for x in jieba.lcut(line) 
                                                                          if x not in stopwords]))
for col in  ['video_tags', 'video_second_class', 'video_director_list', 'video_actor_list']:
    feed_info[col] = feed_info[col].apply(lambda line: ' '.join(re.split(r'[;, ]', line)))


def tfidf_svd(corpus, n_components=16):
    tfidf   = TfidfVectorizer(max_df=0.9, min_df=3, sublinear_tf=True)
    res1     = tfidf.fit_transform(corpus) 
    cnt_vec = CountVectorizer(max_df=0.9, min_df=3)
    res2 = cnt_vec.fit_transform(corpus)
    
    logger.info('svd start')
    svd1     = TruncatedSVD(n_components=n_components, random_state=2021)
    svd_res1 = svd1.fit_transform(res1)
    
    svd2     = TruncatedSVD(n_components=n_components, random_state=2021)
    svd_res2 = svd2.fit_transform(res2)
    
    svd_res = np.concatenate((svd_res1, svd_res2), axis=-1)
    return svd_res


## video_tags, video_second_class
logger.info("开始处理 video_tags")
corpus = feed_info['video_tags'] + ' ' + feed_info['video_second_class']
corpus = [x.strip() for x in corpus.values]

# For lgb
emb_size = 5
emb_res = tfidf_svd(corpus, n_components=emb_size)
fid_tag_svd_emb = pd.concat([pd.DataFrame(feed_info['video_id'].values, columns=['feedid']),
                             pd.DataFrame(emb_res, columns=['fid_tag_svd{}'.format(i) 
                                                             for i in range(2*emb_size)])], axis=1)
fid_tag_svd_emb.to_pickle("../data/features/lgb_emb/fid_tag_svd_emb_lgb.pkl")

## For nn
emb_size = 16
emb_res = tfidf_svd(corpus, n_components=emb_size)
fid_tag_svd_emb = pd.concat([pd.DataFrame(feed_info['video_id'].values, columns=['feedid']),
                             pd.DataFrame(emb_res, columns=['fid_tag_svd{}'.format(i) 
                                                             for i in range(2*emb_size)])], axis=1)
fid_tag_svd_emb.to_pickle("../data/features/lgb_emb/fid_tag_svd_emb_nn.pkl")
logger.info("开始聚类...")
cluster = KMeans(n_clusters=200, random_state=2021, n_jobs=-1)
pred_kmeans = cluster.fit_predict(emb_res)
feed_info['fid_cluster1'] = pred_kmeans


##  ['video_name', 'video_description', 'video_director_list', 'video_actor_list']
logger.info("开始处理video_description")
corpus = feed_info['video_name']+' '+feed_info['video_description']+' '+ \
            feed_info['video_director_list']+' '+feed_info['video_actor_list']
corpus = [x.strip() for x in corpus.values]

# For lgb
emb_size = 5
emb_res = tfidf_svd(corpus, n_components=emb_size)
fid_tag_svd_emb = pd.concat([pd.DataFrame(feed_info['video_id'].values, columns=['feedid']),
                             pd.DataFrame(emb_res, columns=['fid_desc_svd{}'.format(i) 
                                                             for i in range(2*emb_size)])], axis=1)
fid_tag_svd_emb.to_pickle("../data/features/lgb_emb/fid_desc_svd_emb_lgb.pkl")

## For nn
emb_size = 16
emb_res = tfidf_svd(corpus, n_components=emb_size)
fid_tag_svd_emb = pd.concat([pd.DataFrame(feed_info['video_id'].values, columns=['feedid']),
                             pd.DataFrame(emb_res, columns=['fid_desc_svd{}'.format(i) 
                                                             for i in range(2*emb_size)])], axis=1)
fid_tag_svd_emb.to_pickle("../data/features/lgb_emb/fid_desc_svd_emb_nn.pkl")
logger.info("开始聚类...")
cluster = KMeans(n_clusters=200, random_state=2021, n_jobs=-1)
pred_kmeans = cluster.fit_predict(emb_res)
feed_info['fid_cluster2'] = pred_kmeans
## 聚类结果
fid_cluster_df = feed_info[['video_id', 'fid_cluster1', 'fid_cluster2']]
fid_cluster_df.rename(columns={'video_id': 'feedid'}, inplace=True)


## 预处理数据集
logger.info("重新预处理数据集")
train = pd.read_feather("../data/origin/train.feather")
test = pd.read_feather("../data/origin/test.feather")
user_info = pd.read_feather("../data/origin/user_info.feather")
video_info = pd.read_feather("../data/origin/video_info.feather")
logger.info("train.shape {}, test.shape {}, user_info.shape {}, video_info.shape {}".format(train.shape, test.shape, user_info.shape, video_info.shape))

## 改掉列名
train.rename(columns={'user_id': 'userid', 'video_id': 'feedid'}, inplace=True)
test.rename(columns={'user_id': 'userid', 'video_id': 'feedid'}, inplace=True)
user_info.rename(columns={'user_id': 'userid'}, inplace=True)
video_info.rename(columns={'video_id': 'feedid'}, inplace=True)

del user_info['country']   # 删除country特征
test['date_'] = 15
train['date_'] = train['pt_d'].map(dict(zip(list(train['pt_d'].unique()), range(1, 15))))
del train['pt_d']

video_info = video_info.merge(fid_cluster_df, how='left', on=['feedid'])

##  处理video_info表
video_info['video_release_date'] = pd.to_datetime(video_info['video_release_date'])
## 发布的年份
video_info['video_release_year'] = video_info['video_release_date'].apply(lambda x: x.year)
## 发布至今有多少天了
video_info['video_release_ndays'] = pd.to_datetime('20210502') - video_info['video_release_date']
video_info['video_release_ndays'] = video_info['video_release_ndays'].apply(lambda x: x.days)

video_info = video_info[['feedid', 'video_score', 'video_second_class', 'video_duration',
                         'video_release_year', 'video_release_ndays', 'fid_cluster1', 'fid_cluster2']]
video_info['video_class'] = video_info['video_second_class'].fillna('剧情').apply(lambda x: x.split(',')[0].strip())
del video_info['video_second_class']

## 填充均值
for col in ['video_score', 'video_release_year', 'video_release_ndays']:
    tmp = video_info[col].median()
    video_info[col] = video_info[col].fillna(tmp)

video_info['video_class'] = LabelEncoder().fit_transform(video_info['video_class'])
video_info = reduce_mem(video_info)

train.to_feather("../data/origin/train.feather")
test.to_feather("../data/origin/test.feather")
user_info.to_feather("../data/origin/user_info.feather")
video_info.to_feather("../data/origin/video_info.feather")
logger.info("预处理数据完毕...")


logger.info("开始特征工程")

train = pd.read_feather("../data/origin/train.feather")
test = pd.read_feather("../data/origin/test.feather")
user_info = pd.read_feather("../data/origin/user_info.feather")
video_info = pd.read_feather("../data/origin/video_info.feather")

df = pd.concat([train, test], ignore_index=True)
logger.info("df.shape {}".format(df.shape))
del train, test
gc.collect()

## Word2vec特征
logger.info("开始word2vec特征")
user_dict = df.groupby('userid')['feedid'].agg(list)
user_fid_list = user_dict.values.tolist()
logger.info("序列的个数: {}".format(len(user_fid_list)))
logger.info("序列长度为1的个数: {}".format(len([x for x in user_fid_list if len(x) == 1])))


def get_w2v_emb(user_fid_list, emb_size=8, prefix=None):
    ## 训练word2vec 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(user_fid_list, min_count=1, window=10, vector_size=emb_size, seed=2021,
                              sg=1, workers=24, epochs=5)

    ## 将每个feedid的向量保存为pickle
    feed_emb = pd.DataFrame({'feedid': df['feedid'].unique()})
    
    w2v_fid_mat = []
    null_cnt = 0
    for fid in feed_emb.feedid.values:
        try:
            emb = model.wv[fid]
        except:
            emb = np.zeros(emb_size)
            null_cnt += 1
        w2v_fid_mat.append(emb)
    
    logger.info("nullcnt: {}".format(null_cnt))
    w2v_fid_mat = np.array(w2v_fid_mat, dtype=np.float32)
    fid_w2v_emb = pd.concat([feed_emb, pd.DataFrame(w2v_fid_mat, 
                                                    columns=['{}_{}'.format(prefix, i) for i in range(emb_size)])], axis=1)
    fid_w2v_emb['feedid'] = fid_w2v_emb['feedid'].astype(int)
    return fid_w2v_emb

fid_w2v_emb_lgb = get_w2v_emb(user_fid_list, 16, "fid_w2v_emb_lgb")
# fid_w2v_emb_nn = get_w2v_emb(user_fid_list, 64, "fid_w2v_emb_nn")
logger.info("fid_w2v_emb_lgb {}".format(fid_w2v_emb_lgb.shape))

fid_w2v_emb_lgb.to_pickle("../data/features/lgb_emb/fid_w2v_emb_lgb.pkl")
# fid_w2v_emb_nn.to_pickle("../data/features/nn_emb/fid_w2v_emb_nn.pkl")



## 行为序列的tfidf-svd特征
logger.info("行为序列的tfidf-svd特征")
def tfidf_svd(df, f1, f2, lgb_emb_size=8):
    tmp     = df.groupby(f1, as_index=False)[f2].agg({'list': lambda x: ' '.join(list(x.astype('str')))})
    tfidf   = TfidfVectorizer(max_df=0.9, min_df=3, sublinear_tf=True)
    res1     = tfidf.fit_transform(tmp['list']) 
    
    cnt_vec = CountVectorizer(max_df=0.9, min_df=3)
    res2 = cnt_vec.fit_transform(tmp['list'])
    
    
    logger.info('svd start')
    svd1     = TruncatedSVD(n_components=lgb_emb_size, random_state=2021)
    svd_res1 = svd1.fit_transform(res1)
    svd2     = TruncatedSVD(n_components=lgb_emb_size, random_state=2021)
    svd_res2 = svd2.fit_transform(res2)
    svd_res_lgb = np.concatenate((svd_res1, svd_res2), axis=-1)
    logger.info("svd result shape {}".format(svd_res_lgb.shape))
    
    del tmp['list']
    tmp_lgb = tmp[[f1]]
    for i in (range(lgb_emb_size * 2)):
        tmp_lgb['{}_{}_tfidf_svd_lgb_{}'.format(f1, f2, i)] = svd_res_lgb[:, i]
        tmp_lgb['{}_{}_tfidf_svd_lgb_{}'.format(f1, f2, i)] = tmp_lgb['{}_{}_tfidf_svd_lgb_{}'.format(f1, f2, i)].astype(np.float32)
    tmp_lgb[f1] = tmp_lgb[f1].astype(np.int32)
    return tmp_lgb


fid_tfidf_svd_emb_lgb = tfidf_svd(df, 'feedid', 'userid')
uid_tfidf_svd_emb_lgb = tfidf_svd(df, 'userid', 'feedid')

fid_tfidf_svd_emb_lgb.to_pickle("../data/features/lgb_emb/fid_tfidf_svd_emb_lgb.pkl")
uid_tfidf_svd_emb_lgb.to_pickle("../data/features/lgb_emb/uid_tfidf_svd_emb_lgb.pkl")


## 图网络 ProNE特征
logger.info("图网络 ProNE特征")
def get_proNE_embedding(df, col1, col2, emb_size=32):
    ### userid-feedid二部图
    uid_lbl,qid_lbl = LabelEncoder(), LabelEncoder()
    df['new_col1'] = uid_lbl.fit_transform(df[col1])
    df['new_col2'] = qid_lbl.fit_transform(df[col2])
    new_uid_max = df['new_col1'].max() + 1
    df['new_col2'] += new_uid_max
    
    ## 构建图
    G = nx.Graph()
    G.add_edges_from(df[['new_col1','new_col2']].values)

    model = ProNE(G, emb_size=emb_size, n_iter=6, step=12) 
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix,
                             model.step, model.mu, model.theta)
    ## 得到proNE的embedding
    emb = model.transform()

    ## for userid
    uid_emb = emb[emb['nodes'].isin(df['new_col1'])]
    uid_emb['nodes'] = uid_lbl.inverse_transform(uid_emb['nodes'])  # 得到原id
    uid_emb.rename(columns={'nodes' : col1}, inplace=True)
    for col in uid_emb.columns[1:]:
        uid_emb[col] = uid_emb[col].astype(np.float32)
    user_prone_emb = uid_emb[uid_emb.columns]
    user_prone_emb = user_prone_emb.reset_index(drop=True)
    user_prone_emb.columns = [col1] + ['user_prone_emb{}'.format(i) for i in range(emb_size)]
    
    ## for feedid
    fid_emb = emb[emb['nodes'].isin(df['new_col2'])]
    fid_emb['nodes'] = qid_lbl.inverse_transform(fid_emb['nodes'] - new_uid_max)  ## 还原需要减掉
    fid_emb.rename(columns={'nodes' : col2}, inplace=True)
    for col in fid_emb.columns[1:]:
        fid_emb[col] = fid_emb[col].astype(np.float32)
    feed_prone_emb = fid_emb[fid_emb.columns]
    feed_prone_emb = feed_prone_emb.reset_index(drop=True)
    feed_prone_emb.columns = [col2] + ['fid_prone_emb{}'.format(i) for i in range(emb_size)]
    logger.info("user_prone_emb.shape {}, feed_prone_emb.shape {}".format(user_prone_emb.shape, feed_prone_emb.shape))
    return user_prone_emb, feed_prone_emb

user_prone_emb, feed_prone_emb = get_proNE_embedding(df[['userid', 'feedid']],
                                                      col1='userid', col2='feedid', emb_size=32)

user_prone_emb.to_pickle("../data/features/lgb_emb/user_prone_emb_lgb.pkl")
feed_prone_emb.to_pickle("../data/features/lgb_emb/feed_prone_emb_lgb.pkl")


logger.info("开始计算用户的全局兴趣偏好特征...")
def w2v_sent2vec(sentence, fid_2_emb, embed_size):
    """计算句子的平均word2vec向量, sentences是一个句子, 句向量最后会归一化"""
    M = []
    for word in sentence:
        try:
            M.append(fid_2_emb[word])
        except KeyError: # 不在词典里
            continue
    if len(M) == 0:
        return ((-1 / np.sqrt(embed_size)) * np.ones(embed_size)).astype(np.float32)
    M = np.array(M)
    v = M.sum(axis=0)
    return (v / np.sqrt((v ** 2).sum())).astype(np.float32)


def sif_embeddings(sentences, fid_2_emb, fid_2_cnt, emb_size, alpha=1e-3):
#     vlookup = model.wv.vocab  # Gives us access to word index and count
#     vectors = model.wv        # Gives us access to word vectors
#     size = model.vector_size  # Embedding size
    count = 0
    
#     Z = sum([v for k, v in fid_2_cnt.items()])
    # Iterare all words
    output = []
    for s in tqdm(sentences):
        v = np.zeros(emb_size, dtype=np.float32) # Summary vector
        s_cnt = sorted([(x, fid_2_cnt[x]) for x in s],key=lambda x: x[1], reverse=True)
        s = [x[0] for x in s_cnt[:50]]
#         for w in s:
#             # A word must be present in the vocabulary
#             if w in fid_2_emb:
#                 for i in range(emb_size):
#                     v[i] += ( alpha / (alpha + (fid_2_cnt[w] / Z))) * fid_2_emb[w][i]
#                 count += 1
#         if count > 0:
#             for i in range(emb_size):
#                 v[i] *= 1/count
        v = w2v_sent2vec(s, fid_2_emb, emb_size)
        output.append(v)
    return np.vstack(output).astype(np.float32)


## For lgb
fid_tag_emb = pd.read_pickle("../data/features/lgb_emb/fid_tag_svd_emb_lgb.pkl")
fid_desc_emb = pd.read_pickle("../data/features/lgb_emb/fid_desc_svd_emb_lgb.pkl")

fid_2_emb_df = fid_tag_emb.merge(fid_desc_emb, how='left', on=['feedid'])

fid_2_emb = {}
for line in fid_2_emb_df.values:
    fid_2_emb[int(line[0])] = line[1:]
fid_2_cnt = dict(collections.Counter(df['feedid']))


uid_2_fidlist = df.groupby('userid', as_index=False)['feedid'].agg({'fid_list': list})
emb_size = 20
uid_embs = sif_embeddings(uid_2_fidlist['fid_list'].values.tolist(), fid_2_emb, fid_2_cnt, emb_size)

uid_sif_hist_emb_lgb = pd.concat([uid_2_fidlist[['userid']], 
                                  pd.DataFrame(uid_embs, columns=['uid_sif_hist_emb{}'.format(i) 
                                                                  for i in range(emb_size)])], axis=1)
uid_sif_hist_emb_lgb.to_pickle("../data/features/lgb_emb/uid_sif_hist_emb_lgb.pkl")
logger.info("uid_sif_hist_emb_lgb shape {}".format(uid_sif_hist_emb_lgb.shape))
del df
gc.collect()



## 统计特征
## 历史统计特征和CTR特征（历史所有天的）
logger.info("历史统计特征和CTR特征（历史所有天的）")
train = pd.read_feather("../data/origin/train.feather")
test = pd.read_feather("../data/origin/test.feather")
user_info = pd.read_feather("../data/origin/user_info.feather")
video_info = pd.read_feather("../data/origin/video_info.feather")

tmp = pd.get_dummies(train['watch_label'], prefix='watch_label_cls')
del tmp['watch_label_cls_0']
train = pd.concat([train, tmp], axis=1)
del tmp

df = pd.concat([train, test], ignore_index=True)
del train, test
gc.collect()

df = df.merge(user_info, how='left', on=['userid'])
df = df.merge(video_info, how='left', on=['feedid'])
logger.info("df.shape {}".format(df.shape))

## 填充缺失值
for col in ['video_score', 'video_duration', 'video_release_year',
            'video_release_ndays', 'video_class', 'fid_cluster1', 'fid_cluster2']:
    avg = int(df[col].median()) * 1.0
    logger.info("col {}, avg {}".format(col, avg))
    df[col] = df[col].fillna(avg)

## 用户的播放时长
df['watch_playseconds'] = df['watch_label'] * 0.1 * df['video_duration']
df['watch_playseconds'] = df['watch_playseconds'].astype(np.float32)

y_list = ['is_watch', 'is_share', 'is_collect', 'is_comment', 'watch_label', 'watch_playseconds']
y_list = y_list + ['watch_label_cls_{}'.format(i) for i in range(4, 10)]
logger.info("y_list {}, len(y_list) {}".format(y_list, len(y_list)))

## 压缩内存
df = reduce_mem(df)


## 统计历史的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 7
max_day = 15

def get_adjust_df(df):
    for col in df.columns:
        tpe = str(df[col].dtype)
        if tpe[:3] == 'int':
            df[col] = df[col].astype(np.int32)
        if tpe[:5] == 'float':
            df[col] = df[col].astype(np.float32)
    return df

for stat_cols in ([['userid'], ['feedid'], ['age', 'gender'], ['fid_cluster1']]):
    f = '_'.join(stat_cols)
    logger.info('========  {}  ========='.format(f))
    time.sleep(0.5)
    start_time = time.time()
    stat_df = pd.DataFrame()
    for target_day in tqdm(range(2, max_day + 1)):
        left, right = max(target_day - n_day, 1), target_day - 1
        
        tmp = df[(df['date_'] <= right)].reset_index(drop=True)   # 历史
        tmp = tmp[stat_cols + y_list]
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
        tmp = get_adjust_df(tmp)
        
        g = tmp.groupby(stat_cols)
        
        # 特征列
        feats = ['{}_{}day_count'.format(f, n_day)]
        
        for y in y_list:
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
            feats.extend(['{}_{}day_{}_mean'.format(f, n_day, y)])
        tmp.drop_duplicates(stat_cols + ['date_'], inplace=True)
        tmp = tmp[stat_cols + ['date_'] + feats].reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del g, tmp
        gc.collect()
    
    stat_df = reduce_mem(stat_df)
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    del stat_df
    gc.collect()
    logger.info("time costed: {}(s)".format(round(time.time() - start_time, 2)))

df.to_feather("../data/features/df_stat_v1.feather")

del df
gc.collect()


## 历史统计特征和CTR特征（历史3天的）
logger.info("历史统计特征和CTR特征（历史3天的）")
train = pd.read_feather("../data/origin/train.feather")
test = pd.read_feather("../data/origin/test.feather")
user_info = pd.read_feather("../data/origin/user_info.feather")
video_info = pd.read_feather("../data/origin/video_info.feather")

tmp = pd.get_dummies(train['watch_label'], prefix='watch_label_cls')
del tmp['watch_label_cls_0']
train = pd.concat([train, tmp], axis=1)
del tmp

df = pd.concat([train, test], ignore_index=True)
del train, test
gc.collect()

df = df.merge(user_info, how='left', on=['userid'])
df = df.merge(video_info, how='left', on=['feedid'])
logger.info("df.shape {}".format(df.shape))

## 填充缺失值
for col in ['video_score', 'video_duration', 'video_release_year',
            'video_release_ndays', 'video_class', 'fid_cluster1', 'fid_cluster2']:
    avg = int(df[col].median()) * 1.0
    df[col] = df[col].fillna(avg)

## 用户的播放时长
df['watch_playseconds'] = df['watch_label'] * 0.1 * df['video_duration']

## 压缩内存
df = reduce_mem(df)

y_list = ['is_watch', 'is_share', 'is_collect', 'is_comment', 'watch_label', 'watch_playseconds']
y_list = y_list + ['watch_label_cls_{}'.format(i) for i in range(5, 10)]

## 统计历史3天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 3
max_day = 15

def get_adjust_df(df):
    for col in df.columns:
        tpe = str(df[col].dtype)
        if tpe[:3] == 'int':
            df[col] = df[col].astype(np.int32)
        if tpe[:5] == 'float':
            df[col] = df[col].astype(np.float32)
    return df


for stat_cols in ([['userid'], ['feedid']]):
    f = '_'.join(stat_cols)
    logger.info('======== {} ========='.format(f))
    time.sleep(0.5)
    start_time = time.time()
    stat_df = pd.DataFrame()
    for target_day in tqdm(range(2, max_day + 1)):
        left, right = max(target_day - n_day, 1), target_day - 1
        
        tmp = df[(df['date_'] >= left) & (df['date_'] <= right)].reset_index(drop=True)   # 历史
        tmp = tmp[stat_cols + y_list]
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
        tmp = get_adjust_df(tmp)
        
        g = tmp.groupby(stat_cols)
        
        # 特征列
        feats = ['{}_{}day_count'.format(f, n_day)]
        
        for y in y_list:
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
            feats.extend(['{}_{}day_{}_mean'.format(f, n_day, y)])
        tmp.drop_duplicates(stat_cols + ['date_'], inplace=True)
        tmp = tmp[stat_cols + ['date_'] + feats].reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del g, tmp
        gc.collect()
    
    stat_df = reduce_mem(stat_df)
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    del stat_df
    gc.collect()
    logger.info("time costed: {}(s)".format(round(time.time() - start_time, 2)))

df.to_feather("../data/features/df_stat_v2.feather")




logger.info("全局统计特征")
train = pd.read_feather("../data/origin/train.feather")
test = pd.read_feather("../data/origin/test.feather")
user_info = pd.read_feather("../data/origin/user_info.feather")
video_info  = pd.read_feather("../data/origin/video_info.feather")

df = pd.concat([train, test], ignore_index=True)
logger.info("df.shape {}".format(df.shape))

del train, test
gc.collect()

df = df.merge(user_info, how='left', on='userid')
df = df.merge(video_info, how='left', on='feedid')

for col in ['video_score', 'video_duration', 'video_release_year',
            'video_release_ndays', 'video_class', 'fid_cluster1', 'fid_cluster2']:
    avg = int(df[col].median()) * 1.0
    df[col] = df[col].fillna(avg)

## 全局统计（统计count, nunique）
def cnt_stat(df, group_cols, target_col, use_cnt=True, use_nunique=True):
    if isinstance(group_cols, list):
        col_name = '_'.join(group_cols)
    else:
        col_name = 'global_' + group_cols
    if use_cnt:
        df[f'{col_name}_cnt'] = df.groupby(group_cols)[target_col].transform('count')
        df[f'{col_name}_cnt'] = df[f'{col_name}_cnt'].astype(np.float32)
    if use_nunique:
        df[f'{col_name}_dcnt'] = df.groupby(group_cols)[target_col].transform('nunique')
        df[f'{col_name}_dcnt'] =  df[f'{col_name}_dcnt'].astype(np.float32)
    return df

def max_mean_stat(df, group_cols, target_col, use_max=True, use_mean=True, use_sum=False):
    if isinstance(group_cols, list):
        col_name = '_'.join(group_cols)
    else:
        col_name = 'global_' + group_cols
    if use_max:
        df['{}_{}_mean'.format(col_name, target_col)] = df.groupby(group_cols)[target_col].transform('mean')
        df['{}_{}_mean'.format(col_name, target_col)] = df['{}_{}_mean'.format(col_name, target_col)].astype(np.float32)
    if use_mean:
        df['{}_{}_max'.format(col_name, target_col)] = df.groupby(group_cols)[target_col].transform('max')
        df['{}_{}_max'.format(col_name, target_col)] = df['{}_{}_max'.format(col_name, target_col)].astype(np.float32)
    if use_sum:
        df['{}_{}_sum'.format(col_name, target_col)] = df.groupby(group_cols)[target_col].transform('sum')
        df['{}_{}_sum'.format(col_name, target_col)] = df['{}_{}_sum'.format(col_name, target_col)].astype(np.float32)
    return df


df = cnt_stat(df, 'feedid', 'userid')
df = cnt_stat(df, ['feedid', 'date_'], 'userid', use_nunique=False)
df = cnt_stat(df, 'userid', 'feedid')
df = cnt_stat(df, ['userid', 'date_'], 'feedid', use_nunique=False)

df['userid_date_ratio'] = df['userid_date__cnt'] / df['global_userid_cnt']
df['feedid_date_ratio'] = df['feedid_date__cnt'] / df['global_feedid_cnt']

for t in ['userid', 'feedid']:
    df[f'date_first_{t}'] = df.groupby(t)['date_'].transform('min')
    df[f'diff_first_{t}'] = df['date_'] - df[f'date_first_{t}']
    del df[f'date_first_{t}']


df = max_mean_stat(df, 'feedid', 'age', use_max=False)   # 电影的受众年龄
df = max_mean_stat(df, 'feedid', 'gender', use_max=False)   # 电影的受众性别
df = max_mean_stat(df, 'userid', 'video_score')   # 用户的电影评分统计
df = max_mean_stat(df, 'userid', 'video_release_year')   # 用户的电影年份
df = max_mean_stat(df, 'userid', 'video_release_ndays')   # 用户的电影已播映天数

df = reduce_mem(df)
df.to_feather("../data/features/df_stat_v3.feather")




