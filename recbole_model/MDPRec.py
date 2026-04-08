import torch
from torch import nn
from torch.nn import functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
import pickle
import math
import random
import pandas as pd
import copy
from recbole_model.timefeatures import time_features

import numpy as np

class MDPRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MDPRec, self).__init__(config, dataset)
        self.data_name = config["dataset"].split('/')[-1]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.temperature = config["temperature"]
        self.phcl_temperature = config["phcl_temperature"]
        self.phcl_weight = config["phcl_weight"]
        self.beta = config["beta"]


        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
      

        self.item_seq = Encoder(config, self.max_seq_length)
        self.item_seq1 = Encoder(config, self.max_seq_length)
        
        self.item_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.item_lng = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropoutg = nn.Dropout(self.hidden_dropout_prob)

        self.item_lnf = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropoutf = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()

        
        self.cta_moe = Time_Interval_Aware_MoE(config) 
        self.fta_moe = Time_Features_Aware_MoE(config)  
        self.inter_moe = Align_MoE(config)

        self.item_lncta = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropoutcta = nn.Dropout(self.hidden_dropout_prob)

        self.item_lnfta = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropoutfta = nn.Dropout(self.hidden_dropout_prob)

        self.fusion_W1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.fusion_W2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.fusion_W1 = nn.init.xavier_uniform_(self.fusion_W1)
        self.fusion_W2 = nn.init.xavier_uniform_(self.fusion_W2)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_idx, seq_length, timestamp=None):
        item_emb = self.item_embedding(input_idx)
        
        id_pos_emb = self.position_embedding.weight[:input_idx.shape[1]]
        id_pos_emb = id_pos_emb.unsqueeze(0).repeat(item_emb.shape[0], 1, 1)
        item_emb += id_pos_emb
       
        item_emb_g = self.cta_moe(item_emb, timestamp)
        item_emb_g = self.item_lng(self.dropoutg(item_emb_g))    #先drop再归一化，不加item_emb
        
        item_emb_f = self.fta_moe(item_emb, timestamp)
        item_emb_f = self.item_lnf(self.dropoutf(item_emb_f))
        

        item_emb_g = self.item_seq(item_emb_g, output_all_encoded_layers=True)
        item_emb_f = self.item_seq1(item_emb_f, output_all_encoded_layers=True)

        align_info = self.inter_moe(torch.cat([item_emb_g, item_emb_f], dim=-1))
        item_emb_g += align_info[0]
        item_emb_f += align_info[1] # B,L,D

        
        gate_unit = torch.sigmoid(torch.matmul(item_emb_g, self.fusion_W1) + torch.matmul(
            item_emb_f, self.fusion_W2))
        item_seq_full = gate_unit * item_emb_g + (1 - gate_unit) * item_emb_f  # B,L,D
        item_seq_full = self.dropout(self.item_ln(item_seq_full))
        
        item_seq = self.gather_indexes(item_seq_full, seq_length - 1)
        
        
        item_emb_full = self.item_embedding.weight
        item_score = torch.matmul(item_seq, item_emb_full.transpose(0, 1))
        score = item_score
        return item_emb, item_seq, score

    def calculate_loss(self, interaction):
        item_idx = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        timestamp = interaction['timestamp_list']
        item_emb_seq, seq_vectors, score = self.forward(item_idx, item_seq_len, timestamp)
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(score, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        timestamp = interaction['timestamp_list']
        _, _, scores = self.forward(item_seq, item_seq_len, timestamp)
        return scores[:, test_item]

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        timestamp = interaction['timestamp_list']
        _, _, score= self.forward(item_seq, item_seq_len, timestamp)
        return score
    
def calc_reg_loss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret

#FMoE   Top-K
class Time_Features_Aware_MoE(nn.Module):
    def __init__(self,config):
        super(Time_Features_Aware_MoE, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.expert_num = config["timestamp_expert_num"]
        self.time_feature_types = config["time_feature_types"]
        #time_enc = extract_time_feature_type(config["time_feature_types"]) # MHHD :MinuteOfHour HourOfDay
        self.time_dim = len(config["time_feature_types"]) # MHHD 4 

        self.time_enc = nn.Sequential(
                    nn.Linear(self.time_dim, config["c_out"]//config["rda"]),   # t,
                    nn.LayerNorm(config["c_out"]//config["rda"]),
                    nn.ReLU(),
                    nn.Linear(config["c_out"]//config["rda"], config["c_out"]//config["rdb"]),
                    nn.LayerNorm(config["c_out"]//config["rdb"]),
                    nn.ReLU(),
                    
                    UnfoldAndReshape(period=config["period"]),  #B,L//P,D,P
                    
                    nn.Conv2d(in_channels=config["pred_len"]//config["period"],  
                            out_channels=config["pred_len"]//config["period"],  
                            kernel_size=config["ksize"],  
                            padding='same'),
                    
                    FoldBackToSequence(pred_len=config["pred_len"]),
                    nn.Linear(config["c_out"]//config["rdb"], config["c_out"]),
        )
        
        self.gate = nn.Sequential(
            nn.Linear(config["c_out"], config["c_out"] *2), 
            nn.ReLU(), 
            nn.Linear(config["c_out"] *2, self.expert_num))
        
        self.gate_selection = config["timestamp_gate_selection"]
        self.top_k = config["timestamp_top_k"]
        self.expert = [nn.Parameter(torch.Tensor(1, self.hidden_size).to('cuda'), requires_grad=True) for _ in range(self.expert_num)]
        #1,D
        for i in range(self.expert_num):
            nn.init.normal_(self.expert[i], std=0.1)

    def get_time_embedding(self, timestamp):
       
        batch, length = timestamp.shape
        device = timestamp.device
        mask_time = timestamp !=0   # B,L
        timestamp = timestamp.cpu().numpy()
        if self.config["dataset"] in ['lastfm']:
            timestamp = pd.to_datetime(pd.DataFrame(timestamp).values.flatten()/1000, unit='s')
        else:
            timestamp = pd.to_datetime(pd.DataFrame(timestamp).values.flatten(), unit='s')
        time_feature = time_features(timestamp, self.time_feature_types)   
        time_feature = torch.tensor(time_feature, dtype=torch.float32).to(device).transpose(1, 0)
        time_feature = time_feature.reshape(batch,length,self.time_dim) #B,L,t
        time_feature = (mask_time.unsqueeze(-1) * time_feature)   #B,L,1 * B,L,t
        time_emb = self.time_enc(time_feature)  # B,t
        return time_emb
    
    def load_balancing_loss(self, gate_probs, mode='KL'):
        if mode == 'KL':
            expert_avg_probs = gate_probs.mean(dim=0)  # shape: [num_experts]
            uniform_dist = torch.ones_like(expert_avg_probs) / expert_avg_probs.size(0)
            kl_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log(expert_avg_probs),
                uniform_dist
            )
            return kl_loss
        
        elif mode == 'CV':
            importance = gate_probs.sum(dim=0)  # shape: [num_experts]
            importance_mean = importance.mean()
            importance_std = importance.std()
            coeff_vairation = importance_std / (importance_mean)
            return coeff_vairation

        else:
            raise ValueError('Invalid Load Balancing Mode!')

    def forward(self, seq_emb, timestamp):
        time_emb = self.get_time_embedding(timestamp)

        route = F.softmax(self.gate(time_emb), dim=-1) # B,L,DxD,expert_num->B,L,expert_num
        values, topk_indices = torch.topk(route, k=self.top_k, dim=-1)   # B,L,K
        route_weight = torch.zeros_like(route)
        route = route_weight.scatter_(2, topk_indices, values)    #B,L,K

        if self.gate_selection == 'softmax':
            expert_output = []
            for i in range(self.expert_num):
                expert_output.append((seq_emb * self.expert[i]).unsqueeze(2))    #B,L,D * 1,1,D -> B,L,D->B,L,1,D
            
            expert_output = torch.cat(expert_output, dim=2) # B,L,expert_num,D
            expert_proba = torch.sum(expert_output * route.unsqueeze(3), dim=2) # B,L,expert_num,D * B,L,expert_num,1 -> B,L,expert_num,D -> B,L,D
        return expert_proba

class UnfoldAndReshape(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        
    def forward(self, x):
        # x shape: [batch_size, pred_len, features]
        batch_size, pred_len, features = x.shape

        x = x.transpose(1, 2)  # [batch_size, features, pred_len]
        x = x.unfold(dimension=-1, size=self.period, step=self.period)  # B,D,L/P,P

        x = x.transpose(1, 2) #B,L/P,D,P
        new_len = pred_len // self.period
        x = x.reshape(batch_size, new_len, features, self.period)
        return x

class FoldBackToSequence(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len
        
    def forward(self, x):
        # x shape: [batch_size, new_len, features, period]
        batch_size, new_len, features, period = x.shape

        x = x.reshape(batch_size, self.pred_len, features)
        return x

#GMoE
class Time_Interval_Aware_MoE(nn.Module):
    def __init__(self, config):
        super(Time_Interval_Aware_MoE, self).__init__()
        self.data_name = config["dataset"].split('/')[-1]
        self.interval_scale = config["interval_scale"]
        self.hidden_size = config["hidden_size"]
        self.expert_num = config["time_interval_expert_num"]
        self.gate_selection = config["time_interval_gate_selection"]
        #self.gate = nn.Linear(2 * self.hidden_size, self.expert_num)
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size , self.hidden_size *2), 
            nn.ReLU(), 
            nn.Linear(self.hidden_size *2, self.expert_num))
        
        self.absolute_w = nn.Linear(1, self.hidden_size)
        self.absolute_m = nn.Linear(self.hidden_size, self.hidden_size)
        self.top_k = config["time_interval_top_k"]

        self.time_embedding = nn.Embedding(int(self.interval_scale * self.get_interval_num()) + 1, self.hidden_size)

        self.expert = [nn.Parameter(torch.Tensor(1, config["hidden_size"]).to('cuda'), requires_grad=True) for _ in range(self.expert_num)]
        for i in range(self.expert_num):
            nn.init.normal_(self.expert[i], std=0.1)

    def get_interval_num(self):
        with open(f'./dataset/{self.data_name}/interval_num', 'rb') as f: return pickle.load(f)

    # def get_minmax_day(self):
    #     with open(f'./dataset/{self.data_name}/minmax_num', 'rb') as f: return pickle.load(f)

    def get_time_embedding(self, timestamp):
        absolute_embedding = torch.cos(self.freq_enhance_ab(self.absolute_w(timestamp.unsqueeze(2))))
        interval_first = torch.zeros((timestamp.shape[0], 1)).long().to('cuda')
        interval = torch.log2(timestamp[:, 1:] - timestamp[:, :-1] + 1)
        interval_index = torch.floor(self.interval_scale * interval).long()
        interval_index = torch.cat([interval_first, interval_index], dim=-1)
        interval_embedding = self.time_embedding(interval_index)
        return torch.cat([interval_embedding, absolute_embedding], dim=-1)

    def freq_enhance_ab(self, timestamp):
        freq = 10000
        freq_seq = torch.arange(0, self.hidden_size, 1.0, dtype=torch.float).to('cuda')
        inv_freq = 1 / torch.pow(freq, (freq_seq / self.hidden_size)).view(1, -1) # shape = (64)
        return timestamp * inv_freq

    def forward(self, vector, timestamp):
        # 先只实现softmax
        expert_proba = None
        #absolute_embedding = torch.cos(self.freq_enhance_ab(self.absolute_w(timestamp.unsqueeze(2))))
        interval_first = torch.zeros((vector.shape[0], 1)).long().to('cuda')
        interval = torch.log2((timestamp[:, 1:] - timestamp[:, :-1]).clamp(min=0) + 1)
        interval_index = torch.floor(self.interval_scale * interval).long()
        interval_index = torch.cat([interval_first, interval_index], dim=-1)
        interval_embedding = self.time_embedding(interval_index)
        #route = F.softmax(self.gate(torch.cat([interval_embedding, absolute_embedding], dim=-1)), dim=-1)
        route = F.softmax(self.gate(interval_embedding), dim=-1)
        # values, topk_indices = torch.topk(route, k=self.top_k, dim=-1)   # B,L,K
        # route_weight = torch.zeros_like(route)
        # route = route_weight.scatter_(2, topk_indices, values)    #B,L,K

        if self.gate_selection == 'softmax':
            expert_output = []
            for i in range(self.expert_num):
                expert_output.append((vector * self.expert[i]).unsqueeze(2))
            expert_output = torch.cat(expert_output, dim=2)

            expert_proba = torch.sum(expert_output * route.unsqueeze(3), dim=2)
        return expert_proba
#DMoE
class Align_MoE(nn.Module):
    def __init__(self, config):
        super(Align_MoE, self).__init__()
        self.expert_num = config["align_expert_num"]
        self.hidden_size = int(config["hidden_size"])
        self.gate_selection = config["align_gate_selection"]
        self.gate_g = nn.Linear(self.hidden_size, self.expert_num)
        self.gate_f = nn.Linear(self.hidden_size, self.expert_num)
        self.top_k = config["align_gate_top_k"]
        single_expert_net = torch.nn.Sequential(nn.Linear(self.hidden_size *2, self.hidden_size *2),
                                                torch.nn.Dropout(p=0.5),
                                                torch.nn.ReLU(),
                                                nn.Linear(self.hidden_size *2, self.hidden_size *2),
                                                torch.nn.Dropout(p=0.5))
        # nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.expert = nn.ModuleList([single_expert_net for _ in range(self.expert_num)])  # 先实现最简单的专家网络
        self.weight = nn.Parameter(torch.tensor(config["initializer_weight"]).to('cuda'), requires_grad=True)

    def forward(self, vector):  #torch.cat([item_emb, txt_emb, img_emb];   torch.cat([item_emb_g, item_emb_f] #B,L,2D
        # 先只实现softmax
        output = None
        rout_g = F.softmax(self.gate_g(vector[:,:,:self.hidden_size]), dim=-1)  #B,L,K
        values, topk_indices = torch.topk(rout_g, k=self.top_k, dim=-1)   # B,L,K
        route_weight = torch.zeros_like(rout_g)
        rout_g = route_weight.scatter_(2, topk_indices, values)    #B,L,K

        rout_f = F.softmax(self.gate_f(vector[:,:,self.hidden_size:2 * self.hidden_size]), dim=-1)
        # values, topk_indices = torch.topk(rout_f, k=self.top_k, dim=-1)   # B,L,K
        # route_weight = torch.zeros_like(rout_f)
        # rout_f = route_weight.scatter_(2, topk_indices, values)    #B,L,K
        
        if self.gate_selection == 'softmax':
            expert_output = []
            for i in range(self.expert_num):
                expert_output.append(self.expert[i](vector).unsqueeze(2))   #B,L,1,2D
            expert_output = torch.cat(expert_output, dim=2)

            
            output = []
            output.append(self.weight[0] * torch.sum(expert_output[:,:,:,:self.hidden_size] * rout_g.unsqueeze(3), dim=2))
            output.append(self.weight[1] * torch.sum(expert_output[:,:,:, self.hidden_size:2 * self.hidden_size] * rout_f.unsqueeze(3), dim=2))
            
        return output

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class Intermediate(nn.Module):
    def __init__(self, hidden_size=64, hidden_dropout_prob=0.5, hidden_act='gelu'):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size * 4)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        self.dense_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class PeriodEncoder(nn.Module):
    def __init__(self, config, max_seq_length):
        super(PeriodEncoder, self).__init__()

        hidden_size=config["hidden_size"]
        hidden_dropout_prob=config["hidden_dropout_prob"] 
        attn_dropout_prob = config["attn_dropout_prob"]
        
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        init_weight_mat = torch.eye(max_seq_length) * 1.0 + torch.randn(max_seq_length, max_seq_length) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])
       
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(attn_dropout_prob)
        self.dropout1 = nn.Dropout(hidden_dropout_prob)

       
        cycle = max_seq_length
       
        self.a_1 = nn.Parameter(torch.zeros(1, cycle))
        self.a_2 = nn.Parameter(torch.zeros(cycle, 1))
        self.b_1 = nn.Parameter(torch.zeros(1, cycle))
        self.b_2 = nn.Parameter(torch.zeros(cycle, 1))
        self.cycle = cycle

        distance = torch.abs(torch.arange(cycle).unsqueeze(1) - torch.arange(cycle).unsqueeze(0))
        self.diff = nn.Parameter(torch.abs(torch.min(distance%cycle, (-distance)%cycle)).float(),
                                 requires_grad=False)

        self.reset_parameters()
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight_mat, mean=0.0, std=0.10)

    def func(self):
        a = torch.sigmoid((self.a_1@self.a_2))  # 1,C x C,1 -> 1,1
        b = torch.sigmoid((self.b_1@self.b_2)) * self.cycle # 1,C x C,1 -> 1,1
        return 1/(1+torch.exp(a*(self.diff-b))) + torch.exp(-self.diff)/(1+torch.exp(a*b))
    # torch.log(self.func()) 1,1

    def forward(self, input_tensor):
        values = self.v_proj(input_tensor)  # B,L,D
        p_score = torch.log(self.func()) # p, p
        A = F.softplus(self.weight_mat +p_score)  # 1,L,L
        A = self.dropout(A)
        A = F.normalize(A, p=1, dim=-1)  #归一化
        hidden_states = A @ values  #1,L,L->B,L,L x B,L,D 
        hidden_states = input_tensor + self.dropout1(self.out_proj(hidden_states)) 
        hidden_states = self.norm(hidden_states)   #B,L,D
        return hidden_states

class Layer(nn.Module):
    def __init__(self, config, max_seq_length):
        super(Layer, self).__init__()
        hidden_size=config["hidden_size"]
        hidden_dropout_prob=config["hidden_dropout_prob"] 
        hidden_act=config["hidden_act"]
        
        self.enclayer = PeriodEncoder(config, max_seq_length)
        self.intermediate = Intermediate(hidden_size, hidden_dropout_prob, hidden_act)

    def forward(self, hidden_states):
       
        hidden_states = self.enclayer(hidden_states)
        
        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self, config, max_seq_length):
        super(Encoder, self).__init__()

        layer = Layer(config, max_seq_length)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config["n_layers"])])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers[-1]