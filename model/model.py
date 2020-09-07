import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from util.MultiHeadedAttn import *
from model.AGGate import *

def weigth_init(l):
    nn.init.xavier_uniform_(l.weight.data)
class BHC(nn.Module):

    def __init__(self, configs):
        super(BHC, self).__init__()
        self.config = configs
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.pos_layer = torch.nn.Embedding(configs.pos_num, configs.embedding_dim_pos)
        nn.init.xavier_uniform_(self.pos_layer.weight)

        # self.context_gate = Context_Gate(configs)
        # self.context_sents_cnn = Context_CNN(configs, configs.window_size)
        # self.context_gate.initialize()

        # for param in self.bert.parameters():
        #     param.requires_grad = True

        self.context_atten = Context_Attention(configs)

        # self.aggate = AGGate(configs)
        self.aggate = MHAGGate(configs)
        self.in_dim = configs.feat_dim
        if self.config.pos:
            self.in_dim = configs.feat_dim + configs.embedding_dim_pos
            # self.in_dim = configs.hidden_size + configs.embedding_dim_pos
        self.linear = nn.Linear(self.in_dim, 2)
        self.linear.apply(weigth_init)

    def batched_index_select(self, bert_output, bert_clause_b, bert_token_b):
        """
        :param bert_output:         bert_output[0]: (bsize, doc_word_len, dim)
        bert_clause_b: (bsize, doc_len)
        bert_token_b: (bsize, doc_len, seq_len)
        :param bert_clause_b:
        :param bert_token_b:
        :return:
        """
        hidden_state = bert_output[0] #(bsize, doc_word_len, dim)

        dummy_sent = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy_sent)

        return doc_sents_h

        # dummy_token = bert_token_b.reshape(bert_token_b.size(0), -1).unsqueeze(2).expand(bert_token_b.size(0), bert_token_b.size(1), hidden_state.size(2))
        # doc_tokens_h = hidden_state.gather(1, dummy_token)
        # doc_tokens_h = torch.reshape(doc_tokens_h, [bert_token_b.size(0), bert_token_b.size(1), bert_token_b.size(2), -1])
        # return doc_sents_h, doc_tokens_h

    def batched_sent_context_index_select(self, doc_sents_h, context_index):
        """
        :param doc_sents_h: (2, max_doc_len, 768)
        :param context_index: (2, max_doc_len, max_doc_len)
        :return: 还需要一个步骤生成MASK, 来取对应子句的representation
        """
        ##生成对应的mask
        dummy_sents_context = context_index.reshape(context_index.size(0), -1).unsqueeze(2).expand(context_index.size(0), -1, doc_sents_h.size(2))
        context_sents_h = doc_sents_h.gather(1, dummy_sents_context)


        return context_sents_h
    def pos_embed(self, bert_emotion_idx, doc_len):
        max_len = torch.max(doc_len).item()
        batch_size = doc_len.size(0)

        pos_matrix = torch.arange(1, max_len+1).unsqueeze(0).expand(batch_size, -1)
        bert_emotion = bert_emotion_idx.unsqueeze(-1).expand(-1, max_len)

        rp_matrix = pos_matrix - bert_emotion #(2, max_len)
        rp_embed = self.pos_layer((rp_matrix + 69).to(DEVICE)) #(2, max_len, pos_dim)
        if self.config.use_kernel:
            kernel = self.kernel_(rp_matrix, doc_len)
            kernel_ = kernel.unsqueeze(-1)
            rp_embed = kernel_ * rp_embed
            # kernel = self.kernel_generator(rp_matrix)
            # kernel = kernel.unsqueeze(0).expand(batch_size, -1, -1)
            # rp_embed = torch.matmul(kernel, rp_embed)
        return rp_embed

    def kernel_(self, rp, doc_len):
        """
        :param rp: (batch_size, max_len)
        :return: (batch_size, max_len)
        """
        rp_ = rp.type(torch.FloatTensor).to(DEVICE)
        doc_len_ = doc_len.type(torch.FloatTensor).to(DEVICE)
        distri = 1 - torch.abs(rp_) * ((1/doc_len_).unsqueeze(-1))
        return distri
        # return torch.exp(-(torch.pow(rp_, 2))/5.0)

    def kernel_generator(self, rel_pos):
        n_clause = rel_pos.size(1)
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(DEVICE)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_clause, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))

    def forward(self, doc_len, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, bert_token_idx_b, bert_token_lens_b, context_previous_index, context_poster_index, bert_emotion_index):

        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_masks_b.to(DEVICE),
                                token_type_ids=bert_segment_b.to(DEVICE))

        # get sentence representation and token representation
        batch_size = bert_output[0].size(0)
        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE), bert_token_idx_b.to(DEVICE))
        # emotion_sents_h = self.batched_index_select(bert_output, bert_emotion_index.to(DEVICE), bert_token_idx_b.to(DEVICE))
        #####################不考虑文档的层次关系，直接使用[CLS]的方式#############################

        #context_pre_sents_h (2, max_len, max_len, 768)
        context_pre_sents_h = self.batched_sent_context_index_select(doc_sents_h.to(DEVICE), context_previous_index.to(DEVICE))
        # context_pre_sents_h (2, max_len, max_len, 768)

        context_pos_sents_h = self.batched_sent_context_index_select(doc_sents_h.to(DEVICE), context_poster_index.to(DEVICE))



        ################CNN#################
        #context_pre_h (2, max_len, dim) 也就是说每个clause都有一个对应生成的上文向量
        #context_pos_h (2, max_len, dim)
        # context_pre_sents_h = context_pre_sents_h.reshape(-1, context_previous_index.size(2), doc_sents_h.size(2))
        # context_pos_sents_h = context_pos_sents_h.reshape(-1, context_poster_index.size(2), doc_sents_h.size(2))
        # context_pre_sents_h = context_pre_sents_h.unsqueeze(1)
        # context_pos_sents_h = context_pos_sents_h.unsqueeze(1)
        # context_pre_h, context_pos_h = self.context_cnn(context_pre_sents_h, context_pos_sents_h, batch_size)
        #
        # output = self.context_gate(doc_sents_h, context_pre_h, context_pos_h)
        # return output

        ##############Multihead attention##############
        #
        context_pre_sents_h = context_pre_sents_h.reshape(-1, context_previous_index.size(2),
                                                          context_previous_index.size(2), doc_sents_h.size(2))

        context_pos_sents_h = context_pos_sents_h.reshape(-1, context_poster_index.size(2),
                                                          context_poster_index.size(2), doc_sents_h.size(2))

        ###在[cls]上采用multi-head attention
        context_pre_h, context_pos_h =self.context_atten(doc_sents_h, context_pre_sents_h, context_pos_sents_h)

        # out_context = self.context_gate(doc_sents_h, context_pre_h, context_pos_h)
        ##########采用attention guided gate###############
        out_context = self.aggate(doc_sents_h, context_pre_h, context_pos_h)

        if self.config.pos:
            rp_matrix = self.pos_embed(bert_emotion_index, doc_len)
            out_context = torch.cat([out_context, rp_matrix], dim=-1)
        output = self.linear(out_context).squeeze(-1)

        return output

        # return output
        #
        #
        # #####################文档的层次关系，在BERT输出的word representation上操作#############################
        #
        # #context_pre_sents_h (2, max_len, max_len, sen_len, 768)
        # context_pre_tokens_h = self.batched_token_context_index_select(doc_tokens_h, context_previous_index)
        # # context_pre_sents_h (2, max_len, max_len, sen_len, 768)
        # context_pos_tokens = self.batched_token_context_index_select(doc_tokens_h, context_poster_index)
        #
        # #############################CNN############################
        # #context_pre_token_h (2, max_len, max_len, 768)
        # context_pre_token_h = torch.cat(
        #     [self.conv_and_pool(context_pre_sents_h, conv) for i, conv in enumerate(self.convs_tokens_pre)], dim=1)
        # # context_pre_token_h (2, max_len, max_len, 768)
        # context_pos_token_h = torch.cat(
        #     [self.conv_and_pool(context_pos_sents_h, conv) for i, conv in enumerate(self.convs_tokens_pos)], dim=1)
        #
        # #context_pre_h (2, max_len, 768)
        # context_pre_h = torch.cat(
        #     [self.conv_and_pool(context_pre_sents_h, conv) for i, conv in enumerate(self.convs_sents_pre)], dim=1)
        # context_pos_h = torch.cat(
        #     [self.conv_and_pool(context_pos_sents_h, conv) for i, conv in enumerate(self.convs_sents_pos)], dim=1)

    def loss_pre(self, pred_c, y_causes, y_mask):

        y_causes = torch.FloatTensor(y_causes).to(DEVICE)
        pred_c = torch.reshape(pred_c, [-1, 2])
        y_causes = torch.reshape(y_causes, [-1]).long()

        criterion = nn.CrossEntropyLoss()
        #
        # pred = pred_c.masked_select(y_mask)
        # true = y_causes.masked_select(y_mask)
        loss = criterion(pred_c, y_causes)
        return loss

class Context_Gate(nn.Module):
    def __init__(self, configs):
        super(Context_Gate, self).__init__()
        # self.linear_pre_gate = nn.Linear(2 * configs.feat_dim, 1)
        # self.linear_pos_gate = nn.Linear(2 * configs.feat_dim, 1)

        ###应用模型初始化
        # self.linear_pre_gate.apply(weigth_init)
        # self.linear_pos_gate.apply(weigth_init)
        # self.linear_self_gate.apply(weigth_init)

        # self.linear_pre_gate_p = nn.Linear(configs.feat_dim, 1)
        # self.linear_pre_gate_h = nn.Linear(configs.feat_dim, 1)
        #
        # self.linear_pos_gate_p = nn.Linear(configs.feat_dim, 1)
        # self.linear_pos_gate_h = nn.Linear(configs.feat_dim, 1)
        #
        # self.linear_sent_gate_h1 = nn.Linear(configs.feat_dim, 1)
        # self.linear_sent_gate_h2 = nn.Linear(configs.feat_dim, 1)

        self.sigmoid_pre = nn.Sigmoid()
        self.sigmoid_pos = nn.Sigmoid()
        self.sigmoid_sent = nn.Sigmoid()

        self.trans_doc_sents_h = nn.Linear(configs.feat_dim, configs.hidden_size)
        self.trans_pre_sents_h = nn.Linear(configs.feat_dim, configs.hidden_size)
        self.trans_pos_sents_h = nn.Linear(configs.feat_dim, configs.hidden_size)

        self.linear_pre_gate_p = nn.Linear(configs.feat_dim, 1)
        self.linear_pre_gate_h = nn.Linear(configs.feat_dim, 1)

        self.linear_pos_gate_p = nn.Linear(configs.feat_dim, 1)
        self.linear_pos_gate_h = nn.Linear(configs.feat_dim, 1)

        # self.pre_dropout = nn.Dropout(configs.pre_dropout)
        # self.pos_dropout = nn.Dropout(configs.pos_dropout)
        # self.sent_dropout = nn.Dropout(configs.sent_dropout)


    def forward(self, doc_sents_h, context_pre_h, context_pos_h):

        # l_pre = self.sigmoid_pre(self.linear_pre_gate(torch.cat([doc_sents_h, context_pre_h]
        #                                                         , dim=-1)))
        # l_pos = self.sigmoid_pos(self.linear_pos_gate(torch.cat([doc_sents_h, context_pos_h]
        #                                                         , dim=-1)))
        # out_context = (1-l_pre-l_pos)*doc_sents_h + l_pos*context_pos_h \
        #               + l_pre*context_pre_h

        #######################分界线#################################
        # context_pre_h = self.pre_dropout(context_pre_h)
        # context_pos_h = self.pos_dropout(context_pos_h)
        # doc_sents_h = self.sent_dropout(doc_sents_h)
        #
        # l_pre = self.sigmoid_pre(self.linear_pre_gate_p(context_pre_h) +
        #                          self.linear_pre_gate_h(doc_sents_h))
        # l_pos = self.sigmoid_pos(self.linear_pos_gate_p(context_pos_h) +
        #                          self.linear_pos_gate_h(doc_sents_h))
        #
        #
        # out_context = (1-l_pre-l_pos) *doc_sents_h + l_pos*context_pos_h \
        #               + l_pre*context_pre_h

        # ###3#################分界线############################
        # doc_sents_h_ = self.trans_doc_sents_h(doc_sents_h)
        # context_pre_h_ = self.trans_pre_sents_h(context_pre_h)
        # context_pos_h_ = self.trans_pos_sents_h(context_pos_h)
        #
        # l_pre = self.sigmoid_pre(self.linear_pre_gate_p(context_pre_h_) +
        #                          self.linear_pre_gate_h(doc_sents_h_))
        # l_pos = self.sigmoid_pos(self.linear_pos_gate_p(context_pos_h_) +
        #                          self.linear_pos_gate_h(doc_sents_h_))
        # out_context = (1 - l_pos - l_pre) * doc_sents_h_ + l_pos * context_pos_h_ \
        #               + l_pre * context_pre_h_

        ########################分界线#####################################
        # l_pre = self.sigmoid_pre(self.linear_pre_gate_p(context_pre_h) +
        #                          self.linear_pre_gate_h(doc_sents_h))
        # l_pos = self.sigmoid_pos(self.linear_pos_gate_p(context_pos_h) +
        #                          self.linear_pos_gate_h(doc_sents_h))
        # l_sent = self.sigmoid_sent(self.linear_sent_gate_h1(doc_sents_h) +
        #                           self.linear_sent_gate_h2(doc_sents_h))
        #
        # out_context = l_sent * doc_sents_h + l_pos * context_pos_h \
        #               + l_pre * context_pre_h

        ######################分界线############################################
        # ###3#################分界线############################
        doc_sents_h_ = self.trans_doc_sents_h(doc_sents_h)
        context_pre_h_ = self.trans_pre_sents_h(context_pre_h)
        context_pos_h_ = self.trans_pos_sents_h(context_pos_h)

        l_pre = self.sigmoid_pre(self.linear_pre_gate_p(context_pre_h) +
                                 self.linear_pre_gate_h(doc_sents_h))
        l_pos = self.sigmoid_pos(self.linear_pos_gate_p(context_pos_h) +
                                 self.linear_pos_gate_h(doc_sents_h))
        out_context = (1 - l_pos - l_pre) * doc_sents_h_ + l_pos * context_pos_h_ \
                      + l_pre * context_pre_h_
        return  out_context

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)


class Context_CNN(nn.Module):
    def __init__(self, configs, window_size):
        super(Context_CNN, self).__init__()
        self.convs_pre = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, configs.num_filters, (h, configs.feat_dim)))
            for h in window_size])

        self.convs_pos = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, configs.num_filters, (h, configs.feat_dim)))
            for h in window_size])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, context_pre_h, context_pos_h, batch_size):
        context_pre_h = torch.cat(
            [self.conv_and_pool(context_pre_h, conv) for i, conv in enumerate(self.convs_pre)], dim=1)
        context_pos_h = torch.cat(
            [self.conv_and_pool(context_pos_h, conv) for i, conv in enumerate(self.convs_pos)], dim=1)

        context_pre_h = context_pre_h.reshape(batch_size, -1, context_pre_h.size(-1))
        context_pos_h = context_pos_h.reshape(batch_size, -1, context_pos_h.size(-1))

        return context_pre_h, context_pos_h


class Context_Hierarchical_CNN(nn.Module):
    def __init__(self, configs):
        super(Context_Hierarchical_CNN, self).__init__()

        self.context_tokens_cnn = Context_CNN(configs, configs.token_window_size)
        self.context_sents_cnn = Context_CNN(configs, configs.sent_window_size)

    def forward(self, context_pre_tokens_h, context_pos_tokens_h, batch_size):
        context_tokens_pre, context_tokens_pos = self.context_tokens_cnn(context_pre_tokens_h, context_pos_tokens_h, batch_size)
        context_sents_pre, context_sents_pos = self.context_tokens_cnn(context_tokens_pre, context_tokens_pos, batch_size)

        return context_sents_pre, context_sents_pos

class Context_Attention(nn.Module):
    def __init__(self, configs):
        super(Context_Attention, self).__init__()
        self.context_pre_atten = MultiHeaded_Sent_Attention(configs.head_count, configs.model_dim)
        self.context_pos_atten = MultiHeaded_Sent_Attention(configs.head_count, configs.model_dim)

    def forward(self, doc_sents_h, context_pre_h, context_pos_h):
        """
        :param doc_sents_h: (2, max_len, 768)
        :param context_pre_h: (2, max_len, max_len, 768)
        :param context_pos_h: (2, max_len, max_len, 768)
        :return: (2, max_len, 768), (2, max_len, 768)
        """

        context_pre = self.context_pre_atten(context_pre_h, context_pre_h, doc_sents_h)
        context_pos = self.context_pos_atten(context_pos_h, context_pos_h, doc_sents_h)
        return context_pre, context_pos

class Context_Hierarchical_Attention(nn.Module):
    def __init__(self, configs):
        super(Context_Hierarchical_Attention, self).__init__()
        self.context_tokens_atten = Context_Attention(configs)
        self.context_sents_atten = Context_Attention(configs)

    def forward(self, doc_sents_h, context_pre_tokens_h, context_pos_tokens_h):

        context_sents_pre_h, context_sents_pos_h = self.context_tokens_atten(doc_sents_h,
                                                    context_pre_tokens_h, context_pos_tokens_h)

        context_sum_pre_h, context_sum_pos_h = self.context_sents_atten(doc_sents_h,
                                                    context_sents_pre_h, context_sents_pos_h)

        return context_sum_pre_h, context_sum_pos_h







