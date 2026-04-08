import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.data.interaction import Interaction
import torch.nn.functional as F
import math
import copy
import numpy as np

class LRUEmbedding(nn.Module):
    def __init__(self, num_items,config):
        super().__init__()
        
        embed_size = config["bert_hidden_units"]
        bert_dropout = config["bert_dropout"]
        self.token = nn.Embedding(num_items, embed_size, padding_idx=0)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(bert_dropout)

    def get_mask(self, x):
        return (x > 0)  # 0添加在后面

    def forward(self, x):
        mask = self.get_mask(x) #tensor([ True,  True, False, False, False, False, False, False, False, False,.....]
       
        x = self.token(x)
        return self.layer_norm(self.embed_dropout(x)), mask


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)

class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above 

class LRUBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        hidden_size = config["bert_hidden_units"]
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=config["bert_attn_dropout"])
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size*4, dropout=config["bert_dropout"])
    
    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x

class LRUModel(nn.Module):
    def __init__(self, num_items, config):
        super().__init__()
        
        self.hidden_size = config["bert_hidden_units"]
        layers = config["bert_num_blocks"]

        self.lru_blocks = nn.ModuleList([LRUBlock(config) for _ in range(layers)])
        

    def forward(self, x, embedding_weight, mask, labels=None):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]  # B x L x D (64) 默认在序列前面添加0
        return x
        # prediction layer
        # if self.args.dataset_code != 'xlong':
        #     scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias    #B,D 
        #     return scores, None
        # else:
        #     assert labels is not None
        #     if self.training:
        #         num_samples = self.args.negative_sample_size  # 100
        #         samples = torch.randint(1, self.args.num_items+1, size=(*x.shape[:2], num_samples,))
        #         all_items = torch.cat([samples.to(labels.device), labels.unsqueeze(-1)], dim=-1)
        #         sampled_embeddings = embedding_weight[all_items]
        #         scores = torch.einsum('b l d, b l i d -> b l i', x, sampled_embeddings) + self.bias[all_items]
        #         labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
        #         return scores, labels_
        #     else:
        #         num_samples = self.args.xlong_negative_sample_size  # 10000
        #         samples = torch.randint(1, self.args.num_items+1, size=(x.shape[0], num_samples,))  # only one time step
        #         all_items = torch.cat([samples.to(labels.device), labels], dim=-1)
        #         sampled_embeddings = embedding_weight[all_items]
        #         scores = torch.einsum('b l d, b i d -> b l i', x, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
        #         labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
        #         return scores, labels_.reshape(labels.shape)
            

class LRURec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(LRURec, self).__init__(config, dataset)
        # load parameters info
        self.dataset = dataset
        self.config = config
        self.layer_norm_eps = config["layer_norm_eps"]
        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]        

        self.embedding = LRUEmbedding(self.n_items, config)
        self.model = LRUModel(self.n_items, config)
        self.truncated_normal_init()

        self.bias = torch.nn.Parameter(torch.zeros(self.n_items))
       
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    # same as SASRec
    def forward(self, x, labels=None):
        x, mask = self.embedding(x)
        return self.model(x, self.embedding.token.weight, mask, labels=labels)
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq) #返回 B,L,D
        seq_output = self.gather_indexes(seq_output, item_seq_len-1)    # B,D
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.embedding.token(pos_items)
            neg_items_emb = self.embedding.token(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.embedding.token.weight #self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) + self.bias
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq) #返回 B,L,D
        seq_output = self.gather_indexes(seq_output, item_seq_len-1)    # B,D
        test_item_emb = self.embedding.token(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq) #返回 B,L,D
        seq_output = self.gather_indexes(seq_output, item_seq_len-1)    # B,D
        test_items_emb = self.embedding.token.weight #self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores