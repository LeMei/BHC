import sys
sys.path.append('..')
from os.path import join
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import *
from util.util import *
import  random
import os
from main import seed_torch

random.seed(TORCH_SEED)
os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
np.random.seed(TORCH_SEED)
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False


def build_train_data(configs, fold_id, shuffle=True):
    seed_torch()
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=bert_batch_preprocessing, drop_last=True)
    return train_loader


def build_inference_data(configs, fold_id, data_type):
    seed_torch()
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.batch_size,
                                              shuffle=False, collate_fn=bert_batch_preprocessing, drop_last=True)
    return data_loader


class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.split = configs.split

        self.data_type = data_type
        self.train_file = join(data_dir, self.split, TRAIN_FILE % fold_id)
        self.valid_file = join(data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join(data_dir, self.split, TEST_FILE % fold_id)


        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

        self.doc_couples_list,\
        self.y_emotions_list, self.y_causes_list, \
        self.doc_len_list, self.doc_id_list, \
        self.bert_token_idx_list, self.bert_clause_idx_list, self.bert_segments_idx_list, \
        self.bert_token_lens_list, self.context_pre_idx_list, self.context_pos_idx_list, self.bert_token_indices_list, self.bert_emotion_idx_list = self.read_data_file(self.data_type)

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], \
                                            self.y_causes_list[idx]
        bert_emotion_idx = self.bert_emotion_idx_list[idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
        context_pre_idx, context_pos_idx = self.context_pre_idx_list[idx], self.context_pos_idx_list[idx]
        bert_token_indices_idx = self.bert_token_indices_list[idx]

        if bert_token_lens > 512:
            bert_token_idx, bert_clause_idx, \
            bert_segments_idx, bert_token_lens, \
            y_emotions, y_causes, doc_len, context_pre_idx, context_pos_idx, bert_token_indices_idx, bert_emotion_idx = self.token_trunk(bert_token_idx, bert_clause_idx,
                                                                          bert_segments_idx, bert_token_lens, doc_couples,
                                                                        y_emotions, y_causes, doc_len, context_pre_idx, context_pos_idx,
                                                                                               bert_token_indices_idx, bert_emotion_idx)

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)

        assert doc_len == len(y_emotions)
        return y_emotions, y_causes, doc_len, doc_id,\
               bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens, context_pre_idx, context_pos_idx, bert_token_indices_idx, bert_emotion_idx

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        doc_id_list = []
        doc_len_list = []
        y_emotions_list, y_causes_list = [], []
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []
        # bert_emotion_clause_idx_list = []
        bert_emotion_idx_list = []

        doc_couples_list = []

        bert_token_indices_list = []
        context_pre_list = []
        context_pos_list = []

        data_list = read_json(data_file)
        for doc_index, doc in enumerate(data_list):
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_emotions, doc_causes = zip(*doc_couples)
            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)

            bert_emotion_idx_list.append(doc_emotions[0])

            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)

            context_pre_list.append(gen_context(doc_len, False))
            context_pos_list.append(gen_context(doc_len, True))

            y_emotions, y_causes = [], []
            doc_clauses = doc['clauses']
            doc_str = ''
            for i in range(doc_len):
                emotion_label = int(i + 1 in doc_emotions)
                cause_label = int(i + 1 in doc_causes)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)

                clause = doc_clauses[i]
                clause_id = clause['clause_id']
                assert int(clause_id) == i + 1
                doc_str += '[CLS] ' + clause['clause'] + ' [SEP] '

            indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)

            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]

            # bert_emotion_clause_idx_list.append([clause_indices[doc_emotions[0] - 1]])
            doc_token_len = len(indexed_tokens)

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            segments_indices.append(len(indexed_tokens))
            for i in range(len(segments_indices) - 1):
                semgent_len = segments_indices[i + 1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            token_indices = []
            for i in range(len(segments_indices)-1):
                seg_start = segments_indices[i]
                seg_end = segments_indices[i+1]
                token_indices.append(np.arange(seg_start + 1, seg_end).tolist())


            assert len(clause_indices) == doc_len
            assert len(token_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)

            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)

            bert_token_indices_list.append(token_indices)

        return  doc_couples_list, y_emotions_list, y_causes_list, doc_len_list, doc_id_list, \
               bert_token_idx_list, bert_clause_idx_list, bert_segments_idx_list, bert_token_lens_list,\
                context_pre_list, context_pos_list, bert_token_indices_list, \
                bert_emotion_idx_list

    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, y_emotions, y_causes, doc_len, context_pre_idx, context_pos_idx,
                    bert_token_indices_idx, bert_emotion_idx):
        # TODO: cannot handle some extreme cases now
        emotion, cause = doc_couples[0]
        if emotion > doc_len / 2 and cause > doc_len / 2:
            i = 0
            while True:
                temp_bert_token_idx = bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx:]
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]
                    y_emotions = y_emotions[i:]
                    y_causes = y_causes[i:]
                    doc_len = doc_len - i
                    bert_token_lens = len(bert_token_idx)

                    bert_token_indices_idx = bert_token_indices_idx[i:]
                    for c in range(doc_len):
                        bert_token_indices_idx[c] = [w-cls_idx for w in bert_token_indices_idx[c]]

                    context_pre_idx = context_pre_idx[:doc_len]
                    context_pos_idx = context_pos_idx[:doc_len]
                    for c in range(doc_len):
                        context_pre_idx[c] = context_pre_idx[c][:doc_len]
                        context_pos_idx[c] = context_pos_idx[c][:doc_len]


                    break
                i = i + 1
        if emotion < doc_len / 2 and cause < doc_len / 2:
            i = doc_len - 1
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    bert_clause_idx = bert_clause_idx[:i]
                    y_emotions = y_emotions[:i]
                    y_causes = y_causes[:i]

                    bert_token_lens = len(bert_token_idx)

                    bert_token_indices_idx = bert_token_indices_idx[:i]

                    doc_len = i

                    context_pre_idx = context_pre_idx[:doc_len]
                    context_pos_idx = context_pos_idx[:doc_len]
                    for c in range(doc_len):
                        context_pre_idx[c] = context_pre_idx[c][:doc_len]
                        context_pos_idx[c] = context_pos_idx[c][:doc_len]
                    break
                i = i - 1
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               y_emotions, y_causes, doc_len, context_pre_idx, context_pos_idx, \
               bert_token_indices_idx, bert_emotion_idx


def bert_batch_preprocessing(batch):
    y_emotions_b, y_causes_b, doc_len_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b, context_previous, context_poster, bert_indices_b, bert_emotion_b = zip(*batch)

    y_mask_b, y_emotions_b, y_causes_b = pad_docs(doc_len_b, y_emotions_b, y_causes_b)
    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)

    bsz, max_len = bert_token_b.size()
    bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)
    for index, seq_len in enumerate(bert_token_lens_b):
        bert_masks_b[index][:seq_len] = 1

    bert_masks_b = torch.FloatTensor(bert_masks_b)
    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_masks_b.shape

    context_previous = pad_matrix(context_previous)
    context_poster = pad_matrix(context_poster)

    bert_indices_b = pad_clause(bert_indices_b)

    context_previous = torch.LongTensor(context_previous)
    context_poster = torch.LongTensor(context_poster)
    bert_indices_b = torch.LongTensor(bert_indices_b)
    bert_emotion_b = torch.LongTensor(bert_emotion_b)
    doc_len_b = torch.LongTensor(doc_len_b)

    return doc_len_b, \
           np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), doc_id_b, \
           bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_token_lens_b, context_previous, context_poster, bert_indices_b, bert_emotion_b


def pad_clause(doc_indices):
    max_clause_len = max([max([len(s) for s in doc]) for doc in doc_indices])
    max_doc_len = max([len(doc) for doc in doc_indices])
    doc_indices_pad = np.zeros([len(doc_indices), max_doc_len, max_clause_len]).astype(int).tolist()
    for idx, doc in enumerate(doc_indices):
        for idy, clause in enumerate(doc):
            doc_indices_pad[idx][idy][:len(clause)] = clause
    return doc_indices_pad
def pad_matrix(contexts):
    max_len = max([len(s) for s in contexts])
    contexts_pad = np.zeros([len(contexts), max_len, max_len]).astype(int).tolist()
    for idx, doc in enumerate(contexts):
        for idy, clause in enumerate(doc):
            contexts_pad[idx][idy][:len(clause)] = clause

    return contexts_pad

def pad_docs(doc_len_b, y_emotions_b, y_causes_b):
    max_doc_len = max(doc_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_ = [], [], []
    for y_emotions, y_causes in zip(y_emotions_b, y_causes_b):
        y_emotions_ = pad_list(y_emotions, max_doc_len, 0)
        y_causes_ = pad_list(y_causes, max_doc_len, 0)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_emotions_))

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)

    return y_mask_b, y_emotions_b_, y_causes_b_

def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad

def gen_context(doc_len, is_post):
    context = []
    for i in np.arange(doc_len):
        context_tmp = []
        for j in range(i + 1):
            context_tmp.append(j)
        context.append(context_tmp)
    if is_post:
        context.reverse()
    return context