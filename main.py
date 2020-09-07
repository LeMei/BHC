import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
import warnings
from data_loader import *
# from model.model import BHC
# from model.model_hierarchical import BHC
from model.model import BHC
from transformers import AdamW, get_linear_schedule_with_warmup
from util.util import *
from sklearn import metrics
import random
from tensorboardX import SummaryWriter

# torch.cuda.set_device(0)
def seed_torch():
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def main(configs, fold_id):
    seed_torch()
    train_loader = build_train_data(configs, fold_id=fold_id)
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    model = BHC(configs).to(DEVICE)


    params = model.parameters()
    params_bert = model.bert.parameters()
    params_rest = list(model.context_atten.parameters()) + list(model.aggate.parameters()) + \
    list(model.pos_layer.parameters()) + list(model.linear.parameters())

    assert sum([param.nelement() for param in params]) == \
           sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])

    writer = SummaryWriter('log/train_loss')
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if configs.bert_grad == False:
        for param in model.bert.parameters():
            param.requires_grad = False

        _params = list(filter(lambda p: p[1].requires_grad, model.bert.named_parameters()))
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in _params if not any(nd in n for nd in no_decay)],
             'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'eps': configs.adam_epsilon},
            {'params': params_rest,
             'weight_decay': configs.l2}
        ]
    else:
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'eps': configs.adam_epsilon},
            {'params': params_rest,
             'weight_decay': configs.l2}
        ]

    # if configs.discr:
    #     group_unfreeze = ['layer.10., layer.11.']
    #     group_freeze = ['layer.0.']
    #     group_all = ['layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
    #                  'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
    #     for n, p in model.named_parameters():
    #         if any(nd in n for nd in group_freeze):
    #             p.requires_grad = False

    # optimizer_grouped_parameters = [
    #      {'params': [p for n, p in _params if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01, 'eps': configs.adam_epsilon},
    #      {'params': [p for n, p in _params if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0, 'eps': configs.adam_epsilon}
    #      ]
    optimizer = AdamW(params, lr=configs.lr)

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)

    model.zero_grad()
    max_f1, max_p, max_r = -1, None, None

    early_stop_flag = None
    global_step = 0
    for epoch in range(1, configs.epochs+1):
        for train_step, batch in enumerate(train_loader, 1):
            global_step = global_step + 1
            model.train()
            doc_len_b, y_emotions_b, y_causes_b, y_mask_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_token_lens_b, context_pre_b, context_pos_b, bert_indices_b, bert_emotion_b = batch

            output = model(doc_len_b, bert_token_b, bert_segment_b, bert_masks_b,
                                                              bert_clause_b, bert_indices_b, bert_token_lens_b, context_pre_b, context_pos_b, bert_emotion_b)

            loss = model.loss_pre(output, y_causes_b, y_mask_b)
            loss = loss / configs.gradient_accumulation_steps
            writer.add_scalar("train_loss", loss, global_step=global_step)

            loss.backward()
            # optimizer.step()
            # scheduler.step()
            # model.zero_grad()
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if global_step % 50 == 0:
                pred = torch.argmax(output, dim=-1)
                acc, p, r, f1 = _evaluate_prf_binary(y_causes_b, pred, doc_len_b)

                msg = 'Iter: {0:>6}, Train Loss:{1:.6f}, Train Pre:{2:>6.2%}, Train Rec: {3:>6.2%}, Train F1:{4:>6.2%}'
                print(msg.format(global_step, loss, p, r, f1))


        with torch.no_grad():
            model.eval()

            if configs.split == 'split10':
                acc, p, r, f1, loss, pred_r, true_r, doc_len_all, doc_id_all = inference_one_epoch(configs, test_loader, model)
                msg = 'Iter: {0:>6}, Test Loss:{1:.6f}, Test Pre:{2:>6.2%}, Test Rec: {3:>6.2%}, Test F1:{4:>6.2%}'
                print(msg.format(epoch, loss, p, r, f1))

                if f1 > max_f1:
                    early_stop_flag = 1
                    max_f1, max_p, max_r = f1, p, r
                    if configs.debug == True and epoch >= 6:
                        f = open('./2_folder_{}_debug_info.txt'.format(fold_id), 'a+')
                        for idx in range(len(doc_id_all)):
                            doc_len_t = doc_len_all[idx]
                            tmp_p, tmp_t = [], []
                            for idy in range(doc_len_t):
                                tmp_p.append(str(pred_r[idx][idy].item()))
                                tmp_t.append(str(true_r[idx][idy]))


                            f.write(str(epoch) + '\n' )
                            f.write(str(doc_id_all[idx])+'\n')
                            f.write('\t'.join(tmp_t)+'\n')
                            f.write('\t'.join(tmp_p)+'\n')
                        f.close()

                else:
                    early_stop_flag += 1


        if epoch > configs.epochs / 2 and early_stop_flag >= 5:
            break
    return max_p, max_r, max_f1


def inference_one_batch(configs, batch, model):
    doc_len_b, y_emotions_b, y_causes_b, y_mask_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_token_lens_b, context_pre_b, context_pos_b, bert_indices_b, bert_emotion_b = batch
    pred = model(doc_len_b, bert_token_b, bert_segment_b, bert_masks_b,
                   bert_clause_b, bert_indices_b, bert_token_lens_b, context_pre_b, context_pos_b, bert_emotion_b)

    loss = model.loss_pre(pred, y_causes_b, y_mask_b)

    return to_np(loss), pred, y_causes_b, y_mask_b, doc_len_b, doc_id_b


def inference_one_epoch(configs, batches, model):
    loss_all, doc_len_all, pred_all, y_cause_all, y_mask_all, doc_id_all = [], [], [], [],[], []
    for i, batch in enumerate(batches):
        loss, pred, y_cause, y_mask, doc_len_b, doc_id_b = inference_one_batch(configs, batch, model)
        pred = torch.argmax(pred, dim=-1)
        loss_all.append(loss)
        doc_len_all.extend(doc_len_b)
        y_cause_all.extend(y_cause)
        pred_all.extend(pred)
        y_mask_all.extend(y_mask)
        doc_id_all.extend(doc_id_b)
        # print('-------------------'+str(i))

    acc, p, r, f1 = _evaluate_prf_binary(y_cause_all, pred_all, doc_len_all)
    loss = np.mean(np.array(loss_all))
    return acc, p, r, f1, loss, pred_all, y_cause_all, doc_len_all, doc_id_all

def _evaluate_prf_binary(targets, outputs, doc_len):
    """
    :param targets:
    :param outputs:
    :return:
    """
    tmp1, tmp2 = [], []
    for i in range(len(outputs)):
        for j in range(doc_len[i]):
            tmp1.append(outputs[i][j].item())
            tmp2.append(targets[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = metrics.precision_score(y_true, y_pred, average='micro')
    p = metrics.precision_score(y_true, y_pred, average='binary')
    r = metrics.recall_score(y_true, y_pred, average='binary')
    f1 = metrics.f1_score(y_true, y_pred, average='binary')
    return acc, p, r, f1


if __name__ == '__main__':
    configs = Config()
    seed_torch()
    warnings.filterwarnings('ignore')

    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 20
    elif configs.split == 'split20':
        n_folds = 20
        configs.epochs = 15
    else:
        print('Unknown data split.')
        exit()

    P, R, F1 = [], [], []
    for fold_id in range(1, 11):
        print('===== fold {} ====='.format(fold_id))
        seed_torch()
        p, r, f1 = main(configs, fold_id)
        P.append(p)
        R.append(r)
        F1.append(f1)

    print("max_test_pre_avg: {:.4f}, max_test_rec_avg: {:.4f}, max_test_f1_avg: {:.4f}".format(np.mean(P), np.mean(R), np.mean(F1)))

