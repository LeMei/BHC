import torch
import math
import torch.nn as nn
from config import DEVICE
import random
import numpy as np
import os
from config import *

def seed_torch():
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class Attention(nn.Module):
    def __init__(self, configs):
        super(Attention, self).__init__()
        self.configs = configs
        self.linear_1 = nn.Linear(configs.feat_dim, configs.feat_dim)
        self.linear_2 = nn.Linear(configs.feat_dim, configs.feat_dim)
        self.linear_3 = nn.Linear(configs.feat_dim, 1)


        self.sm = nn.Softmax(dim=-1)
        self.tahn = nn.Tanh()

    def forward(self, emotion_h, sents_h):
        """
        :param emotion_h: (2, 1, hidden_dim)
        :param sents_h: (2, doc_len, hidden_dim)
        :return: (2, doc_len, hidden_dim)
        """
        seed_torch()
        doc_len = sents_h.size(1)

        # emotion_q = emotion_h.expand(-1, doc_len, -1)
        emotion_q = self.linear_1(emotion_h)
        sents_q = self.linear_2(sents_h)
        # score = torch.matmul(emotion_h, sents_h.transpose(1,2))
        # score = self.linear_3(emotion_q + sents_q).transpose(1,2)
        score = self.linear_3(self.tahn(emotion_q + sents_q)).transpose(1,2)

        score = self.sm(score)

        return score

class AGGate(nn.Module):
    def __init__(self, configs):
        super(AGGate, self).__init__()
        self.configs = configs
        self.attention = Attention(configs)
        self.linear = nn.Linear(configs.feat_dim, 2)

        self.linear_self_gate = nn.Linear(configs.feat_dim, 1)
        self.sigmoid_self = nn.Sigmoid()

        self.sm = nn.Softmax(dim=2)

    def forward(self, emotion_h, sents_h, pre_sents_h, pos_sents_h):
        """
        :param emotion_h: (2, 1, hidden_dim)
        :param sents_h: (2, doc_len, hidden_dim)
        :return: (2, doc_len, hidden_dim)
        """
        batch_size = sents_h.size(0)
        doc_len = sents_h.size(1)

        contexts_m = torch.ones((batch_size, doc_len, doc_len))
        pre_contexts = torch.tril(contexts_m, -1)
        pos_contexts = torch.triu(contexts_m, 1)
        self_diag = torch.eye(doc_len).expand(batch_size, doc_len, doc_len)

        pre_contexts = pre_contexts.unsqueeze(dim=2)
        pos_contexts = pos_contexts.unsqueeze(dim=2)
        self_diag = self_diag.unsqueeze(dim=2)
        bi_contexts = torch.cat([pre_contexts, self_diag, pos_contexts], dim=2).to(DEVI)

        #(2,4,2,4)
        #bi_contexts_norm = (1 / (torch.sum(bi_contexts, dim=-1) + 1)).unsqueeze(-1)

        score = self.attention(emotion_h, sents_h) #(2,1,4)
        ###score (2, 1, 4)
        score = score.expand(-1, doc_len, -1).unsqueeze(dim=-1)
        ###score (2, 4, 4, 1)

        bi_attn_contexts = torch.matmul(bi_contexts, score)# (2, 4, 3, 1)

        #bi_attn_contexts_norm = bi_attn_contexts * bi_contexts_norm

        # bi_attn_contexts_norm = self.sm(bi_attn_contexts_norm)

        self_attn = 1 - torch.sum(bi_attn_contexts, dim=2) #(2, 4, 1)
        self_score = self.sigmoid_self(self.linear_self_gate(sents_h))

        pre_sents_h = pre_sents_h.unsqueeze(dim=2)
        pos_sents_h = pos_sents_h.unsqueeze(dim=2)
        sents_h_self = sents_h.unsqueeze(dim=2)
        contexts_h = torch.cat([pre_sents_h, sents_h_self, pos_sents_h], dim=2) #(2, 4, 2,4,768)

        sents_contexts_h = torch.sum(bi_attn_contexts * contexts_h, dim=2)
        # sents_contexts_h = torch.sum(bi_attn_contexts * contexts_h, dim=2)
        # sents_contexts_h = sents_h


        # output = self.linear(sents_contexts_h).squeeze(-1)

        return sents_contexts_h
class MHAttention(nn.Module):
    def __init__(self, configs):
        super(MHAttention, self).__init__()
        self.configs = configs
        self.dim_per_head = configs.model_dim // configs.head_count
        self.model_dim = configs.model_dim

        self.head_count = configs.head_count

        self.linear_keys = nn.Linear(configs.model_dim,
                                     configs.head_count * self.dim_per_head)
        self.linear_values = nn.Linear(configs.model_dim,
                                       configs.head_count * self.dim_per_head)
        self.linear_query = nn.Linear(configs.model_dim,
                                      configs.head_count * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(configs.dropout)
        self.final_linear = nn.Linear(configs.model_dim, configs.model_dim)

    def forward(self, key, value, query, mask=None, return_key=False, all_attn=False):
        """
        :param key: (2, 4, 768)
        :param value:(2, 4, 768)
        :param query:(2, 4, 768)
        :param mask:
        :param return_key:
        :param all_attn:
        :return:
        """
        seed_torch()

        batch_size = key.size(0)  # 2
        dim_per_head = self.dim_per_head  # 768/n_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        key_up = shape(self.linear_keys(key))
        value_up = shape(self.linear_values(value))
        query_up = shape(self.linear_query(query))

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))


        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)

        drop_attn = self.dropout(attn)
        # context = unshape(torch.matmul(drop_attn, value_up))
        #
        # output = self.final_linear(context)

        return drop_attn

class MHAGGate(nn.Module):
    def __init__(self, configs):
        super(MHAGGate, self).__init__()
        self.configs = configs
        self.attention = MHAttention(configs)
        self.linear_self_gate = nn.Linear(configs.feat_dim, 1)
        self.sigmoid_self = nn.Sigmoid()

        self.sm = nn.Softmax(dim=2)

    def forward(self, sents_h, pre_sents_h, pos_sents_h):
        """
        :pre_attn: (2, doc_len, doc_len, 1)
        :param sents_h: (2, doc_len, hidden_dim)
        :return: (2, doc_len, hidden_dim)
        """
        seed_torch()
        batch_size = sents_h.size(0)
        doc_len = sents_h.size(1)

        contexts_m = torch.ones((batch_size, doc_len, doc_len))
        pre_contexts = torch.tril(contexts_m, -1)
        pos_contexts = torch.triu(contexts_m, 1)

        pre_contexts = pre_contexts.unsqueeze(dim=2)
        pos_contexts = pos_contexts.unsqueeze(dim=2)
        bi_contexts = torch.cat([pre_contexts, pos_contexts], dim=2).to(DEVICE)
        score = self.attention(sents_h, sents_h, sents_h)  # #(2, 4, 4, 1)
        ###score (2, 1, 4)
        bi_attn_contexts = torch.matmul(bi_contexts, score.transpose(1, 3))
        # bi_attn_contexts = torch.matmul(score.transpose(1, 2), bi_contexts.transpose(2, 3)).transpose(2,3)# (2, 4, 2, 2)

        #bi_attn_contexts_norm = bi_attn_contexts * bi_contexts_norm

        # bi_attn_contexts_norm = self.sm(bi_attn_contexts_norm)

        pre_sents_h = pre_sents_h.unsqueeze(dim=2)
        pos_sents_h = pos_sents_h.unsqueeze(dim=2)
        contexts_h = torch.cat([pre_sents_h, pos_sents_h], dim=2) #(2, 4, 2, 768)

        ##元素乘再汇总
        # sents_contexts_h = torch.sum(bi_attn_contexts * contexts_h, dim=2)

        sents_contexts_h = torch.matmul(bi_attn_contexts.transpose(2,3), contexts_h).squeeze()
        # sents_contexts_h = torch.sum(bi_attn_contexts * contexts_h, dim=2)
        # sents_contexts_h = sents_h



        return sents_contexts_h
















