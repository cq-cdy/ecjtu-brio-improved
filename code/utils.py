from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from typing import List
import torch.nn.functional as F
import torch
from random import Random
import json
r = Random()
'''
得到反义句
'''
def get_antonyms(word):
    words = wn.synsets(word)
    a =  [wl.antonyms() for w_lemmas in [w.lemmas() for w in words] for wl in w_lemmas if wl.antonyms()!=[]]
    if (len(a)> 0):
        return a[0][0].name()
    return''

def get_need_antonyms_words(sentence):

    s = word_tokenize(sentence)
    soure_target_par = []
    for x in s:
        t = get_antonyms(x)
        if(t==''):t = x
        else:soure_target_par.append((x,t))
    return soure_target_par

def get_changed_sentence(sentence:str):
    change_pair = get_need_antonyms_words(sentence)
    for _ in range(len(change_pair)):
        sentence = sentence.replace(change_pair[_][0],change_pair[_][1])
    return  sentence

def batch_get_antonyms_sentences(source_sentences: List[str]) -> List[str] :

    antonyms: List[str]  = []
    for i in range(len(source_sentences)):
        antonyms.append(get_changed_sentence(source_sentences[i]))
    return antonyms

def get_pad_encoded_sentences(encoded_src_sentence:torch.Tensor,encoded_tgt_sentences:torch.Tensor,
                              pad_id) :

        pad_len = max(encoded_src_sentence.shape[-1],encoded_tgt_sentences.shape[-1])
        src_en_pad = F.pad(input=encoded_src_sentence,pad=(0,
                pad_len-encoded_src_sentence.shape[-1]),value=pad_id)
        tgt_en_pad = F.pad(input=encoded_tgt_sentences,pad=(0,
                pad_len-encoded_tgt_sentences.shape[-1]),value=pad_id)

        return src_en_pad,tgt_en_pad


def get_probability_flabel(real_label: list):
    for i in range(len(real_label)):
        if (r.random() > 0.6):
            real_label[i] = get_changed_sentence(real_label[i])
    return real_label

def json_write(file_name,dict_):
    with open(file_name, "w") as f:
        json.dump(dict_, f)
        f.close()

def json_read(file_name):

    with open(file_name, 'r') as f:
        load_dict = json.load(f)
        f.close()
    return load_dict


import torch


def get_best_candidate_index(rouges: list, bs: list):
    count = [{i: rouges.count(i)} for i in set(rouges)]
    count.sort(key=lambda x: list(x.values())[0], reverse=True)

    if (len(count) == 1):  # 说明rouge 1，2，l全是同一句
        return list(count[0].keys())[0]

    elif (len(count) == 2):

        r1 = list(count[0].keys())[0]
        r2 = list(count[1].keys())[0]
        if r1 == bs[0] or r2 == bs[0]:
            return bs[0]
        else:
            return r1

    elif (len(count) == 3):  # 最大rouge 1，2， l的句子都不一样就用bs
        return bs[0]

    else:
        raise 'need debug!'


def get_sorted_candidates(item_json: dict):
    new_candidates = []

    # item_json = utils.json_read(f'{os.path.join(root_path,i)}')

    candidates = item_json['candidates']
    rouge = item_json['rouge']
    bs = item_json['bs']

    while (len(candidates) > 0):

        rouge_1_f = []
        # rouge_1_r = []
        rouge_2_f = []
        # rouge_2_r = []
        rouge_l_f = []
        # rouge_l_r = []

        for i in rouge:
            rouge_1_f.append(i['rouge-1']['f'])
            # rouge_1_r.append(i['rouge-1']['r'])
            rouge_2_f.append(i['rouge-2']['f'])
            # rouge_2_r.append(i['rouge-2']['r'])
            rouge_l_f.append(i['rouge-l']['f'])
            # rouge_l_r.append(i['rouge-l']['r'])

        rouge_index = [torch.Tensor(rouge_1_f).argmax(-1).item(),
                       # torch.Tensor(rouge_1_r).argmax(-1).item(),
                       torch.Tensor(rouge_2_f).argmax(-1).item(),
                       # torch.Tensor(rouge_2_r).argmax(-1).item(),
                       torch.Tensor(rouge_l_f).argmax(-1).item(),
                       # torch.Tensor(rouge_l_r).argmax(-1).item()
                       ]

        bs_index = [torch.Tensor(bs['f1']).argmax(-1).item(),
                    # torch.Tensor(bs['recall']).argmax(-1).item()
                    ]

        index = get_best_candidate_index(rouge_index, bs_index)
        #print(rouge_index, bs_index, '->', index)
        new_candidates.append(candidates[index].replace('\n', ' ').strip())

        del candidates[index]
        del rouge[index]
        del bs['f1'][index]
        del bs['recall'][index]

    return new_candidates

