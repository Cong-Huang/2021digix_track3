# coding=UTF-8

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import *
from collections import defaultdict
import gc, pickle, os, time
from utils import reduce_mem, uAUC, get_logger, fast_auc

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers.interaction import FM, BiInteractionPooling
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
from deepctr_torch.layers import DNN, concat_fun, InteractingLayer

import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader,RandomSampler, SequentialSampler, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
from collections import Counter
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pd.set_option('display.max_columns', None)



logger = get_logger("../data/log/nn_train_log.txt")

logger.info("开始读取数据")
train = pd.read_feather("../data/model_data/train_lgb1.feather")
valid_14 = pd.read_feather("../data/model_data/valid_14.feather")
test = pd.read_feather("../data/model_data/test.feather")
logger.info("train, valid, test shape: {}, {}, {}".format(train.shape, valid_14.shape, test.shape))


'''
## For dubug
train = train.sample(n=200000).reset_index(drop=True)
valid_14 = valid_14.sample(n=100000).reset_index(drop=True)
test = test.reset_index(drop=True)
logger.info("train, valid, test shape: {}, {}, {}".format(train.shape, valid_14.shape, test.shape))
'''


df = pd.concat([train, valid_14, test], ignore_index=True)
del train, valid_14, test
gc.collect()

df['video_release_ndays'] = df['video_release_ndays'].apply(lambda x: max(x, 0))
df['global_userid_video_release_ndays_mean'] = df['global_userid_video_release_ndays_mean'].apply(lambda x: max(x, 0))
df['global_userid_video_release_ndays_max'] = df['global_userid_video_release_ndays_max'].apply(lambda x: max(x, 0))

logger.info("get word2id...")
def get_word2id(df, col, is_del=True):
    cnt_dict0 = dict(Counter(df[col]))
    if is_del:
        cnt_dict = {k: v for k, v in cnt_dict0.items() if v >= 2}
        word2id = {k: (i+2) for i, k in enumerate(cnt_dict.keys())}
    else:
        word2id = {k: i for i, k in enumerate(cnt_dict0.keys())}
    logger.info("{}, {} -> {}".format(col, len(cnt_dict0), len(word2id)))
    return word2id

userid_2_id = get_word2id(df, 'userid', is_del=False)
feedid_2_id = get_word2id(df, 'feedid', is_del=False)


logger.info("开始读取userid和feedid的embedding...")
uid_all_emb = pd.read_feather("../data/features/uid_all_emb_lgb.feather")
fid_all_emb = pd.read_feather("../data/features/fid_all_emb_lgb.feather")
logger.info("uid_all_emb.shape {}, fid_all_emb.shape{}".format(uid_all_emb.shape, fid_all_emb.shape))
uid_emb_size, fid_emb_size = uid_all_emb.shape[1]-1, fid_all_emb.shape[1]-1

all_userid = set(df['userid'].unique())
all_feedid = set(df['feedid'].unique())
logger.info("len all userid {}, len all feedid {}".format(len(all_userid), len(all_feedid)))
uid_2_emb, fid_2_emb = {}, {}
for line in uid_all_emb.values:
    uid = int(line[0])
    if uid in all_userid:
        uid_2_emb[uid] = line[1:].astype(np.float32)

for line in fid_all_emb.values:
    fid = int(line[0])
    if fid in all_feedid:
        fid_2_emb[fid] = line[1:].astype(np.float32)

logger.info("len uid_2_emb {}, len fid_2_emb {}".format(len(uid_2_emb), len(fid_2_emb)))
del uid_all_emb, fid_all_emb
gc.collect()


sparse_features = ['userid', 'feedid', 'age', 'gender', 'province',
                   'city', 'city_level', 'device_name', 'video_class', 'video_release_year',
                   'fid_cluster1', 'fid_cluster2']
y_list = ['is_share', 'watch_label']
dense_features = [col for col in df.columns if col not in sparse_features + y_list + ['date_']]
logger.info("len sparse fea, len dense fea {}, {}".format(len(sparse_features), len(dense_features)))
time.sleep(0.5)

logger.info("对类别特征进行label_encoder，数值特征归一化")
## 类别特征
for col in tqdm(sparse_features[2:]):
    lbl = LabelEncoder()
    df[col] = lbl.fit_transform(df[col])
    df[col] = df[col].astype(np.int32)

## 数值特征
for col in tqdm(dense_features[:97]):
    x = df[col].astype(np.float64)
    x = np.log(x + 1.0)
    mms = MinMaxScaler()
    x = mms.fit_transform(x.values.reshape(-1, 1))
    df[col] = x.reshape(-1).astype(np.float16)
    df[col] = df[col].fillna(0.0)

train = df[df['date_'] <= 13].reset_index(drop=True)
valid_14 = df[df['date_'] == 14].reset_index(drop=True)
test = df[df['date_'] == 15].reset_index(drop=True)
logger.info("train, valid, test shape: {}, {}, {}".format(train.shape, valid_14.shape, test.shape))

del df
gc.collect()


# DNN作为主编码器
class NN_Model(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,
                 use_fm=True,
                 embed_dim=32,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0.0, init_std=0.001, seed=1024,
                 dnn_dropout=0.5, dnn_activation='relu', dnn_use_bn=True,
                 task='binary', device='cpu', gpus=None,
                 class_nums=2,
                 ):
        super(NN_Model, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm

        if use_fm:
            self.fm = BiInteractionPooling()

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns) + embed_dim, dnn_hidden_units,
                       activation=dnn_activation,
                       l2_reg=l2_reg_dnn,
                       dropout_rate=dnn_dropout,
                       use_bn=dnn_use_bn, init_std=init_std, device=device)

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], class_nums).to(device)
        self.to(device)


    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)   # [bs, n, emb_dim]
            fm_out = self.fm(fm_input)   # [bs, 1, emb_dim]
            fm_out = fm_out.squeeze(1)   # [bs, emb_dim]

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if self.use_fm and len(sparse_embedding_list) > 0:
            dnn_input = torch.cat([fm_out, dnn_input], dim=-1)   # [bs, x + embed_dim]

        dnn_output = self.dnn(dnn_input)

        y_logit = self.dnn_linear(dnn_output)
        return y_logit

# 打印模型参数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def predict(model, test_loader, device):
    model.eval()
    pred = []
    time.sleep(0.5)
    with torch.no_grad():
        for x in (test_loader):
            y_pred = model(x[0].to(device))
            y_pred = F.softmax(y_pred, dim=-1)
            y_pred = y_pred.cpu().data.numpy()
            pred.append(y_pred)
    pred = np.concatenate(pred, axis=0)   #[n, class_nums]
    return pred

def onehot_encode(nums, k):
    res = np.zeros((len(nums), k))
    for i, x in (enumerate(nums)):
        res[i, int(x)] = 1
    res = res.astype(np.int32)
    return res


def calc_weighted_auc(watch_y_true, valid_pred):
    watch_y_pred = valid_pred
    watch_y_true = onehot_encode(watch_y_true, 10)
    assert watch_y_true.shape == watch_y_pred.shape
    auc_list = []
    for i in range(1, 10):
        score = fast_auc(watch_y_true[:, i], watch_y_pred[:, i])
        auc_list.append(score)
    y2_auc = sum(np.array(auc_list) * np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]))

    weighted_auc = y2_auc / 3.0
    return round(weighted_auc, 4), list(np.round(auc_list, 5))


def evaluate(model, valid_loader, valid_y_true, device):
    valid_y_pred = predict(model, valid_loader, device)
    res = calc_weighted_auc(valid_y_true, valid_y_pred)
    return res


def train_model(model, train_loader, valid_loader, valid_y_true,
                optimizer, epochs, device, model_save_file, best_score=0.0):
    train_bs = len(train_loader)

    patience = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        logger.info("======= epoch {} ======".format(epoch+1))
        model.train()
        start_time = time.time()
        total_loss_sum = 0
        time.sleep(1.0)
        for idx, (out) in tqdm(enumerate(train_loader)):
            y_pred = model(out[0].to(device))
            y = out[1].to(device)

            #loss = F.binary_cross_entropy(y_pred.squeeze(1), y)
            loss = criterion(y_pred, y.long().squeeze())

            reg_loss = model.get_regularization_loss()
            total_loss = loss + reg_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            total_loss_sum += total_loss.item()

            if (idx + 1) == train_bs:
                time.sleep(0.5)
                LR = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info("Epoch {:03d} | Step {:04d} / {} | Loss {:.4f} | Reg Loss {:.4f}| LR {:.5f} | Time {:.4f}".format(
                            epoch+1, idx+1, train_bs, total_loss_sum/(idx+1), reg_loss.item(), LR, time.time() - start_time))

        time.sleep(0.5)
        if valid_y_true.max() == 1:
            y_pred = predict(model, valid_loader, device)
            y_pred = y_pred[:, 1]
            assert valid_y_true.shape == y_pred.shape
            score = roc_auc_score(valid_y_true, y_pred)
        else:
            score, auc_list = evaluate(model, valid_loader, valid_y_true, device)
            logger.info("auc list {}".format(auc_list))

        logger.info("Epoch:{} 结束，验证集AUC = {}".format(epoch + 1, score))
        if score > best_score:
            best_score = score
            patience = 0
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save.state_dict(),  "../data/save_model/{}".format(model_save_file))
            logger.info("save model finished!")
        else:
            patience += 1

        logger.info("Valid cur AUC = {}, Valid best AUC = {}, Cost Time {:.2f}".format(score, best_score,
                                                                              time.time() - start_time))
        if patience >= 2:
            logger.info("Early Stopped! ")
            break
    return best_score


class MyDataset(Dataset):
    def __init__(self, df, sparse_cols, dense_cols, labels,
                 uid_2_id, fid_2_id, uid_2_emb, fid_2_emb,
                 uid_emb_size, fid_emb_size):
        self.sparse_features = df[sparse_cols].values
        self.dense_features = df[dense_cols].values
        self.dates = df['date_'].values
        self.labels = df[labels].values

        self.uid_2_id = uid_2_id
        self.fid_2_id = fid_2_id
        self.uid_2_emb = uid_2_emb
        self.fid_2_emb = fid_2_emb

        self.uid_emb_size = uid_emb_size
        self.fid_emb_size = fid_emb_size

        self.df_len = df.shape[0]

    def __len__(self):
        return self.df_len

    def __getitem__(self, i):
        # 标签信息，日期信息
        label = [self.labels[i]]
        # Sparse特征
        sparse_f = [int(x) for x in self.sparse_features[i]]
        # Dense特征
        dense_f = list(self.dense_features[i])

        # Embedding特征
        uid, fid = sparse_f[0], sparse_f[1]
        all_emb_f = list(self.uid_2_emb.get(uid, [0.0]*self.uid_emb_size))
        all_emb_f.extend(list(self.fid_2_emb.get(fid, [0.0]*self.fid_emb_size)))

        ## Sparse
        sparse_f[0] = self.uid_2_id.get(uid, 1)
        sparse_f[1] = self.fid_2_id.get(fid, 1)

        return (
                torch.FloatTensor(sparse_f + dense_f + all_emb_f),
                torch.FloatTensor(label),
               )


def get_loader(df, y_label, batch_size=1024*4, train_mode=True):
    if train_mode:
        ds = MyDataset(df, sparse_features, dense_features, labels=y_label,
                       uid_2_id=userid_2_id, fid_2_id=feedid_2_id,
                       uid_2_emb=uid_2_emb, fid_2_emb=fid_2_emb,
                       uid_emb_size=uid_emb_size, fid_emb_size=fid_emb_size)

        loader = Data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=True,
                                 num_workers=30,
                                 pin_memory=True)
    else:
        ds = MyDataset(df, sparse_features, dense_features, labels=y_label,
                       uid_2_id=userid_2_id, fid_2_id=feedid_2_id,
                       uid_2_emb=uid_2_emb, fid_2_emb=fid_2_emb,
                       uid_emb_size=uid_emb_size, fid_emb_size=fid_emb_size)
        loader = Data.DataLoader(dataset=ds, batch_size=batch_size*4, shuffle=False,
                                 num_workers=30,
                                 pin_memory=True)
    return loader


vocab_size_list = [len(userid_2_id)+2, len(feedid_2_id)+2]
vocab_size_list.extend([max(train[feat].max(), valid_14[feat].max(), test[feat].max())+1 for feat in sparse_features[2:]])
logger.info("vocab_size list {}".format(vocab_size_list))

emb_size = 48
actions = ['is_share', 'watch_label']
# count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, vocab_size_list[i], embedding_dim=emb_size)
                          for i, feat in enumerate(sparse_features)] +\
                         [DenseFeat(feat, 1) for feat in dense_features +\
                         ['emb_{}'.format(i) for i in range(uid_emb_size + fid_emb_size)]]
logger.info("fixlen feature columns nums: {}".format(len(fixlen_feature_columns)))

# 所有特征列， dnn和linear都一样
dnn_feature_columns = fixlen_feature_columns    # for DNN
linear_feature_columns = fixlen_feature_columns   # for Embedding
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)   # all-特征名字
logger.info("feature nums is {}".format(len(feature_names)))


pickle.dump([sparse_features, dense_features, userid_2_id, feedid_2_id, uid_2_emb, fid_2_emb,
             uid_emb_size, fid_emb_size, fixlen_feature_columns], open("../data/model_data/nn_model_components.pkl", 'wb'))
pickle.dump(valid_14, open("../data/model_data/nn_model_valid_fea.pkl", 'wb'))
pickle.dump(test, open("../data/model_data/nn_model_test_fea.pkl", 'wb'))


# DEVICE
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    logger.info('*** cuda ready ***')
    device = 'cuda:0'

logger.info("开始训练......")

for action in actions:
    logger.info("******** {} ********".format(action))
    train_loader = get_loader(train, action, train_mode=True)
    valid_loader = get_loader(valid_14, action, train_mode=False)
    test_loader = get_loader(test, action, train_mode=False)
    logger.info("len train {}, valid {}, test loader: {}".format(len(train_loader),  len(valid_loader), len(test_loader)))
    # 定义模型
    if action == 'watch_label':
        class_nums = 10
    else:
        class_nums = 2
    model = NN_Model(linear_feature_columns=linear_feature_columns,
                     dnn_feature_columns=dnn_feature_columns,
                     embed_dim=emb_size,
                     use_fm=True,
                     dnn_use_bn=True,
                     dnn_hidden_units=(2048, 1024, 512, 256),
                     init_std=0.001, dnn_dropout=0.5, task='binary',
                     l2_reg_embedding=1e-5,
                     l2_reg_linear=1e-5,
                     l2_reg_dnn=0.0,
                     device=device,
                     class_nums=class_nums)
    model.to(device)
    logger.info(get_parameter_number(model))

    ## 优化器和训练模型
    if action == 'is_share':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=0.003)
    
    

    model_save_file = 'best_model_{}.bin'.format(action)
    num_epochs = 10
    logger.info("样本标签分布: \n{}".format(valid_14[action].value_counts()))
    ## 训练开始
    best_score = train_model(model, train_loader, valid_loader, valid_14[action],
                             optimizer, epochs=num_epochs, device=device, model_save_file=model_save_file)

    ## 小学习率继续训练
    model.load_state_dict(torch.load("../data/save_model/{}".format(model_save_file)))
    
    if action == 'is_share':
        optimizer = optim.Adagrad(model.parameters(), lr=4e-5)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=5e-5)

    train_model(model, train_loader, valid_loader, valid_14[action],
                optimizer, epochs=num_epochs, device=device, model_save_file=model_save_file, best_score=best_score)

    del test_loader, train_loader, valid_loader
    gc.collect()


# ## 保存数据
# valid_pred_nn = valid_14[['userid', 'feedid', 'is_share', 'watch_label']]
# valid_pred_nn['is_share_pred'] = valid_preds[0][:, 1]
# valid_pred_nn = pd.concat([valid_pred_nn, pd.DataFrame(valid_preds[1][:, 1:],
#                                                        columns=['watch_label_pred_{}'.format(i) for i in range(1, 10)])], axis=1)
# test_pred_nn = test[['userid', 'feedid']]
# test_pred_nn['is_share_pred'] = test_preds[0][:, 1]
# test_pred_nn = pd.concat([test_pred_nn, pd.DataFrame(test_preds[1][:, 1:],
#                                                      columns=['watch_label_pred_{}'.format(i) for i in range(1, 10)])], axis=1)


# logger.info("保存推断的数据")
# valid_pred_nn.to_feather("../data/submit/valid_pred_nn.feather")
# test_pred_nn.to_feather("../data/submit/test_pred_nn.feather")
