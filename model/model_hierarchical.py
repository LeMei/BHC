import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from util.MultiHeadedAttn import *

class BHC(nn.Module):

    def __init__(self, configs):
        super(BHC, self).__init__()
        self.config = configs
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)

        self.context_gate = Context_Gate(configs)
        # self.context_sents_cnn = Context_CNN(configs, configs.window_size)

        self.context_atten = Context_Sent_Attention(configs)

        self.context_hierarchical_atten = Context_Hierarchical_Attention(configs)

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


        dummy_token = bert_token_b.reshape(bert_token_b.size(0), -1).unsqueeze(2).expand(bert_token_b.size(0), bert_token_b.size(1) * bert_token_b.size(2), hidden_state.size(2))
        doc_tokens_h = hidden_state.gather(1, dummy_token)
        doc_tokens_h = torch.reshape(doc_tokens_h, [bert_token_b.size(0), bert_token_b.size(1), bert_token_b.size(2), -1])
        return doc_sents_h, doc_tokens_h

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

    def is_reasonable_tensor(self, x, token_len, is_sents):
        if not is_sents:
            x_dummy = torch.where(x > token_len[0])
            num = x_dummy[0].size(0)
            return num == 0

        seq_len = x.size(1)
        x_dummy = torch.where(x>seq_len)
        num = x_dummy[0].size(0)
        return num==0



    def batched_token_context_index_select(self, doc_tokens_h, sents_context_index, sents2token_context_index):
        """
        :param doc_sents_h: (2, max_doc_len, 768)
        :param context_index: (2, max_doc_len, max_doc_len, max_sen_len)
        :return: 还需要一个步骤生成MASK, 来取对应子句的representation
        """
        ##生成对应的mask

        #参数中的context_index是在sents级别上的, 需要将其转换为tokens级别上
        def build_token_context_index(sents_index, sents2token_index):
            """
            :param sents_index: (2, 17, 17)
            :param sents2token_index: (2, 17, 23)
            :return: (2, 17, 17, 23)
            """

            # sents2token_index_dummy = sents2token_index.unsqueeze(2).expand(sents2token_index.size(0),
            #                                                 sents2token_index.size(1), sents_index.size(2), sents2token_index.size(2))
            # sents_index_dummy = sents_index.unsqueeze(3).expand(sents_index.size(0),
            #                                                 sents_index.size(1), sents_index.size(2), sents2token_index.size(2))
            # context_tokens_index = sents2token_index_dummy.gather(2, sents_index_dummy)

            # sents_index_dummy = sents_index.unsqueeze(2)
            # sents_index_dummy = torch.cat([sents_index] * sents2token_index.size(1), dim=1) #(2, 17, 17, 17)
            # sents2token_index_dummy = torch.cat([sents2token_index] * sents_index.size(2), dim=1) #(2, 17, 17, 23)
            #
            # context_tokens_index = torch.matmul(sents_index_dummy, sents2token_index_dummy)

            sents_index_dummy = sents_index.view(-1)
            sents2token_index_dummy = sents2token_index.view(-1, sents2token_index.size(-1))

            context_tokens_index = sents2token_index_dummy[sents_index_dummy]
            context_tokens_index = context_tokens_index.view(sents_index.size(0), sents_index.size(1),
                                                             sents2token_index.size(1), sents2token_index.size(2))

            return context_tokens_index


        # dummy_tokens_context = context_index.reshape(context_index.size(0), -1).unsqueeze(2).expand(context_index.size(0), -1, doc_tokens_h.size(2))
        context_tokens_index = build_token_context_index(sents_context_index.to(DEVICE), sents2token_context_index.to(DEVICE))
        context_tokens_index_dummy = context_tokens_index.unsqueeze(4).expand(context_tokens_index.size(0), context_tokens_index.size(1),
                                                            context_tokens_index.size(2), context_tokens_index.size(3), doc_tokens_h.size(-1))
        context_tokens_h = doc_tokens_h.unsqueeze(1).expand(context_tokens_index.size(0), context_tokens_index.size(1),
                                                            context_tokens_index.size(2), context_tokens_index.size(3), doc_tokens_h.size(-1)).gather(1, context_tokens_index_dummy)
        return context_tokens_h

    def forward(self, doc_len, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, bert_token_idx_b, bert_token_lens_b, context_previous_index, context_poster_index):

        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_masks_b.to(DEVICE),
                                token_type_ids=bert_segment_b.to(DEVICE))

        # get sentence representation and token representation
        batch_size = bert_output[0].size(0)
        doc_sents_h, doc_tokens_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE), bert_token_idx_b.to(DEVICE))

        #####################不考虑文档的层次关系，直接使用[CLS]的方式#############################

        #context_pre_sents_h (2, max_len, max_len, 768)
        # context_pre_sents_h = self.batched_sent_context_index_select(doc_sents_h.to(DEVICE), context_previous_index.to(DEVICE))
        # # context_pre_sents_h (2, max_len, max_len, 768)
        #
        # context_pos_sents_h = self.batched_sent_context_index_select(doc_sents_h.to(DEVICE), context_poster_index.to(DEVICE))


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
        # context_pre_sents_h = context_pre_sents_h.reshape(-1, context_previous_index.size(2),
        #                                                   context_previous_index.size(2), doc_sents_h.size(2))
        #
        # context_pos_sents_h = context_pos_sents_h.reshape(-1, context_poster_index.size(2),
        #                                                   context_poster_index.size(2), doc_sents_h.size(2))
        #
        # context_pre_h, context_pos_h =self.context_atten(doc_sents_h, context_pre_sents_h, context_pos_sents_h)
        #
        # output = self.context_gate(doc_sents_h, context_pre_h, context_pos_h)
        # return output

        #
        #
        # #####################文档的层次关系，在BERT输出的word representation上操作#############################
        # doc_tokens_h (2, max_len, max_sen_len, 768)
        # #context_pre_sents_h (2, max_len, max_len, sen_len, 768)
        ###对previous index和 bert_token_index做一个合理性判断
        if not self.is_reasonable_tensor(context_previous_index, bert_token_lens_b, is_sents=True):
            print(context_previous_index)
        if not self.is_reasonable_tensor(context_poster_index, bert_token_lens_b, is_sents=True):
            print(context_poster_index)

        if not self.is_reasonable_tensor(bert_token_idx_b, bert_token_lens_b, is_sents=False):
            print(bert_token_idx_b)
        context_pre_tokens_h = self.batched_token_context_index_select(doc_tokens_h.to(DEVICE), context_previous_index.to(DEVICE), bert_token_idx_b.to(DEVICE))
        # # context_pre_sents_h (2, max_len, max_len, sen_len, 768)
        context_pos_tokens_h = self.batched_token_context_index_select(doc_tokens_h.to(DEVICE), context_poster_index.to(DEVICE), bert_token_idx_b.to(DEVICE))

        ##########################基于token表示生成clause表示，根据clause表示生成上文表示和下文表示
        # 将token
        # context_pre_tokens_h = torch.reshape(context_pre_tokens_h, [-1, context_pre_tokens_h.size[]])

        context_pre_summery_h, context_pos_summery_h = self.context_hierarchical_atten(doc_sents_h, context_pre_tokens_h,
                                                              context_pos_tokens_h)

        output = self.context_gate(doc_sents_h, context_pre_summery_h, context_pos_summery_h)
        return output
        #
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
        self.linear_pre_gate = nn.Linear(2 * configs.feat_dim, 1)
        self.linear_pos_gate = nn.Linear(2 * configs.feat_dim, 1)
        self.linear = nn.Linear(configs.feat_dim, 2)

        self.sigmoid_pre = nn.Sigmoid()
        self.sigmoid_pos = nn.Sigmoid()

    def forward(self, doc_sents_h, context_pre_h, context_pos_h):

        l_pre = self.sigmoid_pre(self.linear_pre_gate(torch.cat([doc_sents_h, context_pre_h]
                                                                , dim=-1)))
        l_pos = self.sigmoid_pos(self.linear_pos_gate(torch.cat([doc_sents_h, context_pos_h]
                                                                , dim=-1)))
        out_context = doc_sents_h + l_pos*context_pos_h \
                      + l_pre*context_pre_h

        output = self.linear(out_context).squeeze(-1)
        return  output


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

class Context_Token_Attention(nn.Module):
    def __init__(self, configs):
        super(Context_Token_Attention, self).__init__()
        self.context_pre_atten = MultiHeaded_Token_Attention(configs.head_count, configs.model_dim)
        self.context_pos_atten = MultiHeaded_Token_Attention(configs.head_count, configs.model_dim)

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

class Context_Sent_Attention(nn.Module):
    def __init__(self, configs):
        super(Context_Sent_Attention, self).__init__()
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
        self.context_tokens_atten = Context_Token_Attention(configs)
        self.context_sents_atten = Context_Sent_Attention(configs)

    def forward(self, doc_sents_h, context_pre_tokens_h, context_pos_tokens_h):

        context_sents_pre_h, context_sents_pos_h = self.context_tokens_atten(doc_sents_h,
                                                    context_pre_tokens_h, context_pos_tokens_h)

        # context_sents_pre_h = context_sents_pre_h.reshape(-1, context_previous_index.size(2),
        #                                                   context_previous_index.size(2), doc_sents_h.size(2))
        #
        # context_sents_pos_h = context_sents_pos_h.reshape(-1, context_poster_index.size(2),
        #                                                   context_poster_index.size(2), doc_sents_h.size(2))

        context_sum_pre_h, context_sum_pos_h = self.context_sents_atten(doc_sents_h,
                                                    context_sents_pre_h, context_sents_pos_h)

        return context_sum_pre_h, context_sum_pos_h







