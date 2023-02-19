import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import  torch.nn.functional as F

# def lp_loss(all_prob,candidate_id,cand_mask):
#     real_prob=  torch.mul(all_prob,cand_mask.unsqueeze(-1))
#     lp_score = torch.gather(real_prob, 3, candidate_id).squeeze(-1).contiguous()

#     # lp_loss
#     mask = torch.ones_like(lp_score)
#     mask[lp_score == 0] = 0
#     for i in range(1,lp_score.size(-1)):
#         lp_score[:,:,i] =  lp_score[:,:,i-1] + (lp_score[:,:,i-1]/(1+lp_score[:,:,i-1]))
#     lp_score = torch.mul(lp_score,mask)
#     lp_loss = torch.mean(torch.sum(lp_score,dim=-1)/torch.sum(mask,dim=-1))
#     return lp_loss

def acc_rankling_loss(all_prob,candidate_id,cand_mask,pad_token_id = 0,margin = 0.001,lenth_penalty = 2.0,is_lpsum = True):
    loss_func = torch.nn.MarginRankingLoss(0.0)
    real_prob=  torch.mul(all_prob,cand_mask.unsqueeze(-1))
    max_index = torch.argmax(real_prob,dim=-1)
    
    score = torch.gather(real_prob, 3, max_index.unsqueeze(-1)).squeeze(-1)
    label_score = torch.gather(all_prob, 3, candidate_id).squeeze(-1)
    
    #############lp_loss#######################
    lp_score =  - label_score.contiguous()
    # mask = torch.ones_like(lp_score,device = real_prob.device)
    # mask[lp_score == 0] = 0
    for i in range(1,lp_score.size(-1)):
        x =  lp_score[:,:,i-1].clone()
        lp_score[:,:,i] =  x + (x/(1+x))
    lp_score = torch.mul(lp_score,cand_mask)
    if(is_lpsum):
        lp_loss = torch.mean(torch.sum(lp_score,dim=-1)/torch.sum(cand_mask,dim=-1))
    else:
        index = torch.sum(cand_mask,dim = -1)
        index = index.type(torch.int64) -1
        last_score = torch.gather(lp_score,2,index.unsqueeze(-1)).squeeze(-1)
        lp_loss = torch.sum(last_score / torch.sum(cand_mask,-1),dim = -1).mean()   
    #######################################
    
    score = torch.mul(score, cand_mask).sum(-1) / (
                    (cand_mask.sum(-1) + 0) ** lenth_penalty)  # [bz, cand_num]
    label_score = torch.mul(label_score, cand_mask).sum(-1) / (
                    (cand_mask.sum(-1) + 0) ** lenth_penalty)  # [bz, cand_num]
    ones = torch.ones_like(score, device=score.device)
    TotalLoss = loss_func(score, score, ones)
    n = label_score.size(1)
    for i in range(1, n):
        
        pos_score = score[:, :-i]
        neg_score = score[:, i:]
        neg_score = neg_score.contiguous().view(-1)
        pos_score = pos_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(margin * i)
        loss = loss_func(pos_score, neg_score, ones)
        TotalLoss += loss
    acc = torch.mean((torch.mul((max_index[:,0,:] == candidate_id.squeeze(-1)[:,0,:]),cand_mask[:,0,:]).sum(-1)) / cand_mask[:,0,:].sum(-1))
    return TotalLoss,acc,lp_loss

class label_smooth_loss(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(label_smooth_loss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, input, target):
        input = input.transpose(1, 2) # [batch_size, seq_len, word_num]
        input = torch.log_softmax(input, dim=2)
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(input) * self.epsilon * 1 / k
        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()
        return loss

class CELoss(nn.Module):

    def __init__(self,
                 epsilon = 0.1,
                 reduction = 'mean',weigth = None):
        super(CELoss,self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weigth
    def reduce_loss(self,loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self,i,j):
        return (1-self.epsilon) * i + self.epsilon *j

    def forward(self,predict_tensor,target):
        assert  0 <= self.epsilon <1
        if self.weight is not None:
            self.weight = self  .weight.to(predict_tensor.device)
        num_classes = predict_tensor.size(-1)
        log_preds = F.log_softmax(predict_tensor,dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim = -1))
        negative_log_likelihood_loss = F.nll_loss(
            log_preds,target,reduction=self.reduction,weight=self.weight
        )
        return self.linear_combination(negative_log_likelihood_loss,loss/num_classes)
    

def TripletLoss(score, summary_score=None, margin=0.001, gold_margin=0, gold_weight=0, no_gold=False, no_cand=False):

    zeros = torch.zeros_like(score,device=score.device)
    loss_func = torch.nn.TripletMarginLoss(0.0)
    TotalLoss = loss_func(anchor = zeros,positive = score, negative = score)
    # candidate loss
    N = score.size(1)
    if not no_cand:
        for i in range(1, N):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous()
            neg_score = neg_score.contiguous()
            zeros = torch.zeros_like(pos_score)
            loss_func = torch.nn.TripletMarginLoss(margin * i)
            loss = loss_func(anchor = zeros,positive = pos_score, negative = neg_score)
            #print(loss)
            TotalLoss += loss

    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous()
    neg_score = neg_score.contiguous()
    zeros = torch.zeros_like(pos_score)
    loss_func = torch.nn.TripletMarginLoss(gold_margin)
    gold_loss = loss_func(anchor = zeros,positive = pos_score, negative = neg_score)
    #print('gold')
    #print(gold_loss)
    TotalLoss += gold_weight * gold_loss
    #print('ok')
    return TotalLoss


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=0, no_gold=False, no_cand=False):

    ones = torch.ones_like(score,device=score.device)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss

    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    gold_loss = loss_func(pos_score, neg_score, ones)
    TotalLoss += gold_weight * gold_loss
    return TotalLoss

# def TripletLoss(anchors,positives,negatives,margin = 0,trib_margin_weight = 1.1):

#     total_loss = 0
#     n = anchors.size(1)
#     pos_score = positives.contiguous().view(-1,positives.shape[-1])
#     neg_score = negatives.contiguous().view(-1,negatives.shape[-1])
#     for  i in range(n):
#         loss_func = torch.nn.TripletMarginLoss((i+1) * trib_margin_weight)
#         anchor =  anchors[:,i].contiguous().view(-1,positives.shape[-1])
#         loss = loss_func(anchor = anchor, positive = pos_score, negative = neg_score)
#         total_loss +=loss

#     return total_loss / (anchors.size(1) * anchors.size(0) * anchors.size(2))
# def TripletLoss(score,summary_score,margin = 0,trib_margin_weight = 1.1):

#     total_loss = 0
#     n = score.size(1)
#     pos_score = summary_score
#     neg_score = score[:,-1]
#     for  i in range(n-1):
#         loss_func = torch.nn.TripletMarginLoss(i * trib_margin_weight)
#         anchor =  score[:,i].contiguous().unsqueeze(-1)
#         positive = pos_score.contiguous().unsqueeze(-1)
#         negative = neg_score.contiguous().unsqueeze(-1)
#         loss = loss_func(anchor = anchor, positive = positive, negative = negative)
#         total_loss +=loss
#     return total_loss / (n-1)