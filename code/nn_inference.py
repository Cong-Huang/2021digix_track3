# coding=UTF-8

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import *
import gc, pickle, os, time
from utils import reduce_mem, uAUC, get_logger, fast_auc

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers.interaction import FM, BiInteractionPooling
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
pd.set_option('display.max_columns', None)



logger = get_logger("../data/log/nn_inference.txt")
logger.info("开始读取数据")

[sparse_features, dense_features, userid_2_id, feedid_2_id, uid_2_emb, fid_2_emb,
    uid_emb_size, fid_emb_size, fixlen_feature_columns] = pickle.load(open("../data/model_data/nn_model_components.pkl", 'rb'))
valid_14 = pickle.load(open("../data/model_data/nn_model_valid_fea.pkl", 'rb'))
test = pickle.load(open("../data/model_data/nn_model_test_fea.pkl", 'rb'))


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
        for x in tqdm(test_loader):
            y_pred = model(x[0].to(device))
            y_pred = F.softmax(y_pred, dim=-1)
            y_pred = y_pred.cpu().data.numpy()
            pred.append(y_pred)
    pred = np.concatenate(pred, axis=0)   #[n, class_nums]
    return pred



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



# 所有特征列， dnn和linear都一样
dnn_feature_columns = fixlen_feature_columns    # for DNN
linear_feature_columns = fixlen_feature_columns   # for Embedding
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)   # all-特征名字
logger.info("feature nums is {}".format(len(feature_names)))



# DEVICE
emb_size=48
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    logger.info('*** cuda ready ***')
    device = 'cuda:0'

logger.info("开始预测......")

valid_preds = []
test_preds = []

for action in ['is_share', 'watch_label']:
    logger.info("******** {} ********".format(action))
    valid_loader = get_loader(valid_14, action, train_mode=False)
    test_loader = get_loader(test, action, train_mode=False)
    logger.info("len valid loader {}, test loader: {}".format(len(valid_loader), len(test_loader)))
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

    model_save_file = 'best_model_{}.bin'.format(action)
    num_epochs = 10
    logger.info("样本标签分布: \n{}".format(valid_14[action].value_counts()))

    ## 加载模型权重
    model.load_state_dict(torch.load("../data/save_model/{}".format(model_save_file)))
    
    pred = predict(model, valid_loader, device)
    valid_preds.append(pred)

    pred = predict(model, test_loader, device)
    test_preds.append(pred)

    del test_loader, valid_loader
    gc.collect()


# ## 保存数据
valid_pred_nn = valid_14[['userid', 'feedid', 'is_share', 'watch_label']]
valid_pred_nn['is_share_pred'] = valid_preds[0][:, 1]
valid_pred_nn = pd.concat([valid_pred_nn, pd.DataFrame(valid_preds[1][:, 1:],
                                                       columns=['watch_label_pred_{}'.format(i) for i in range(1, 10)])], axis=1)
test_pred_nn = test[['userid', 'feedid']]
test_pred_nn['is_share_pred'] = test_preds[0][:, 1]
test_pred_nn = pd.concat([test_pred_nn, pd.DataFrame(test_preds[1][:, 1:],
                                                     columns=['watch_label_pred_{}'.format(i) for i in range(1, 10)])], axis=1)


logger.info("保存推断的数据")
valid_pred_nn.to_feather("../data/submit/valid_pred_nn_1.feather")
test_pred_nn.to_feather("../data/submit/test_pred_nn_1.feather")


