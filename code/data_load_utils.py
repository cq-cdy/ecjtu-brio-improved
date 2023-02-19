import  os,json

from torch.utils.data import Dataset

from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
import utils
from random import Random
r = Random()
class BrioDataset(Dataset):
    def __init__(self, fdir , tokenizer , max_len=80,k_sort = 4 ,is_test=False, total_len=512, is_sorted=True, max_num=16,
                 is_untok=True, is_pegasus=True, num=-1):
        """ data format: article, abstract, [(candidiate_i, score_i,bs_i)] """
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            d  = [i for i in os.listdir(fdir) if i.endswith('.json')]
            self.num = len(d)
            # if num > 0:
            #     self.num = min(len(os.listdir(fdir)), num)
            # else:
            #     self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)
        self.tok = tokenizer
        self.maxlen = max_len
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = max_num
        self.is_untok = is_untok
        self.is_pegasus = is_pegasus
        self.k_sort = k_sort

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
            #with open(os.path.join(self.fdir, "12.json"), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        if self.is_untok:
            article = data["article_untok"]
        else:
            article = data["article"]
        src_txt = " ".join(article)
        src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)
        if self.is_untok:
            abstract = data["abstract_untok"]
        else:
            abstract = data["abstract"]

        if self.maxnum > 0:
            candidates = data["candidates_untok"][:self.maxnum]
            _candidates = data["candidates"][:self.maxnum]
            data["candidates"] = _candidates

        if self.sorted:
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)

            _candidates = sorted(_candidates, key=lambda x:x[1], reverse=True)
            #candidates = sort_with_bs(candidates,k=self.k_sort,bs_index=2)
            data["candidates"] = _candidates

        cand_txt = [" ".join(abstract)] + [" ".join(x[0]) for x in candidates]
        #for i in range( - self.k_sort - 1 ,0 ): cand_txt[i] = utils.get_changed_sentence(cand_txt[i])

        cand = self.tok.batch_encode_plus(cand_txt, max_length=self.maxlen, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)
        candidate_ids = cand["input_ids"]
        if self.is_pegasus:
            # add start token
            _candidate_ids = candidate_ids.new_zeros(candidate_ids.size(0), candidate_ids.size(1) + 1)
            _candidate_ids[:, 1:] = candidate_ids.clone()
            _candidate_ids[:, 0] = self.tok.pad_token_id
            candidate_ids = _candidate_ids

        result = {
            "src_input_ids": src_input_ids,
            "candidate_ids": candidate_ids,
            }
        if self.is_test:
            result["data"] = data

        return result

def sort_with_bs(cand,k = 4 ,bs_index = 2):

    s_sort = []
    l = len(cand)
    k_ = l // k
    low = up = 0
    while( up<= len(cand)):

        up = low+k_
        end = min(l,up)
        if (low == end):break
        tmp = cand[low:end]
        tmp.sort(key=lambda x:x[bs_index]['f1'],reverse=True)
        s_sort.extend(tmp)
        #print(low,'->',end)
        low = up
    return s_sort

class ValSet(Dataset):
    def __init__(self,file_path,tok,args,length = -1):
        self.args = args
        self.file_path = file_path
        self.d  = [i for i in os.listdir(file_path) if i.endswith('.json')]
        self.tok = tok
        self.l = length


    def __getitem__(self, idx):
        with open(os.path.join(self.file_path, "%d.json"%idx), "r") as f:
            data = json.load(f)
        src_txt = data['x']
        src = self.tok.batch_encode_plus([src_txt], max_length=self.args.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)

        abstract = data["label"]
        label= self.tok.batch_encode_plus([abstract], max_length=self.args.max_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)
        label_id = label["input_ids"]
        if self.args.is_pegasus:
            # add start token
            _label_id = label_id.new_zeros(label_id.size(0), label_id.size(1) + 1)
            _label_id[:, 1:] = label_id.clone()
            _label_id[:, 0] = self.tok.pad_token_id
            label_id = _label_id
        result = {
            "src_input_ids": src_input_ids,
            "candidate_ids": label_id,
            }

        return result
    def __len__(self):
        return self.l

def collate_mp_brio(batch, pad_token_id, is_test=False):

    global data
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)

    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
        }
    if is_test:
        result["data"] = data
    return result