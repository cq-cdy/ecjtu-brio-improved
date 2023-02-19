import torch
class Config():

    def __init__(self,data_name:str):

        assert data_name in ['xsum','cnndam']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epoch  = 10
        self.cache_dir = '../cache'
        self.margin = 0.001
        self.gold_margin = 0
        self.gold_weight = 0
        self.acc_rank_weight = 0
        self.mle_weight = 0.1
        self.contraste_weight = 10
        self.lp_weight = 0.005
        self.warmup_steps = 10000
        self.grad_norm = 0
        self.seed = 970903
        self.max_lr = 2e-3
        self.max_num = 16
        self.smooth = 0.1
        self.adding = 0.0
        self.score_mode = 'log'
        self.normalize = True
        self.is_lpsum = False if data_name == 'xsum' else True
        self.is_pegasus = True if data_name == 'xsum' else False
        self.val_size = 25 if data_name == 'xsum' else 15
        self.stand_rouge = 0.54839 if data_name == 'xsum' else 0.4503139
        self.candidates_train_dir = './xsum/diverse/train' if data_name == 'xsum' else  './cnndm/diverse/train'
        self.batch_size = 4 if data_name == 'xsum' else 4
        self.contraste_weight = 10 if data_name == 'xsum' else 10
        self.acc_rank_weight = 0
        self.accumulate_step = 4 if data_name == 'xsum' else 6
        self.scale = 0.01 if data_name == 'xsum' else 1
        self.max_len = 80 if data_name == 'xsum' else 120
        self.total_len = 512 if data_name == 'xsum' else 1024
        self.length_penalty = 0.6 if data_name == 'xsum' else 2.0
        self.gen_max_len = 62 if data_name == 'xsum' else 140
        self.gen_min_len = 11 if data_name == 'xsum' else 55