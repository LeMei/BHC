import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
# torch.cuda.set_device(0)
# DEVICE = torch.device('cuda:0')
TORCH_SEED = 129
DATA_DIR = './data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE  = 'fold%s_test.json'

# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'

        self.n_split = 10

        self.bert_cache_path = 'bert-base-chinese'
        self.feat_dim = 768
        self.hidden_size= 200

        self.embedding_dim_pos = 128
        self.pos_num = 138
        self.pos = True

        self.window_size = [2, 3]
        self.num_filters = self.feat_dim // len(self.window_size)

        self.epochs = 10
        self.lr = 2e-5
        self.batch_size = 2
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8

        self.discr = True

        self.head_count = 1
        self.model_dim = 768

        self.save = True
        self.save_path = './BHC.pkl'

        self.dropout = 0.1
        self.pre_dropout = 0.1
        self.pos_dropout = 0.1
        self.sent_dropout = 0.1

        self.debug = False
        self.use_kernel = False

        self.bert_grad = True

