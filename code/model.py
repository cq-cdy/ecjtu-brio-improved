import torch

from torch import nn
import SELECT_MODEL_AND_TOK
import torch.nn.functional as F

class Imporved_BRIO(nn.Module):

    def __init__(self, model_name, dataset_name,args):
        super(Imporved_BRIO,self).__init__()
        self.model, self.tokenizer \
            = SELECT_MODEL_AND_TOK.get_model_tokenizer(model_name, dataset_name,args=args)
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, text_id, candidate_id, normalize=True, score_mode="base", length_penalty=1, require_gold=True, adding=0):
        
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        cand_mask = candidate_id != self.pad_token_id
        cand_mask[:, :, 0] = 1
        output = self.model(
            input_ids=text_id, 
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=True
            )

        output = output[0]  # [bz x cand_num, seq_len, word_dim]
        output = output.view(batch_size, -1, output.size(1), output.size(2)) # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]
        output = output[:, :, :-1]  # truncate last token
        candidate_id = candidate_id[:, :, 1:]  # shift right
        cand_mask = candidate_id != self.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)
        if normalize:
            if score_mode == "log":
                _output = F.log_softmax(output, dim=3)
            else:
                _output = F.softmax(output, dim=3)
            all_prob = _output
            scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        else:
            scores = torch.gather(output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        cand_mask = cand_mask.float()
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1) + adding) ** length_penalty) # [bz, cand_num]
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs}
        else:
            output = {'score': scores, "probs": probs}
        return output,candidate_id,cand_mask,all_prob
