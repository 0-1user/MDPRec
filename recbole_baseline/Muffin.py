
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.loss import EmbLoss
from recbole.model.abstract_recommender import SequentialRecommender
import copy
import math

    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, 'silu':F.silu}


class Muffin(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"] 
        self.inner_size = config["inner_size"]  
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]

        self.lfm_encoder = LFMEncoder(config)
        self.gfm_encoder = GFMEncoder(config)
        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.concat_layer = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.kernel_size= config['kernel_size']

        # UAF
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size ,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

    
    def sequence_mask(self, input_ids):
        mask = (input_ids != 0) * 1
        return mask.unsqueeze(-1) 
    
    def make_embedding(self, sequence, seq_mask):
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = item_embeddings
        item_embeddings *= seq_mask
        item_embeddings = self.LayerNorm(item_embeddings)
        item_embeddings = self.dropout(item_embeddings)
        return item_embeddings
    
    def forward(self, input_ids, item_seq_len):
        seq_mask = self.sequence_mask(input_ids)
        sequence_emb = self.make_embedding(input_ids, seq_mask)

        # UAF
        frequency_emb = torch.fft.rfft(sequence_emb, dim=1,norm='ortho')
        filter = torch.sigmoid(self.freq_conv_encoder(frequency_emb.abs().permute(0,2,1)))
        
        # GFM
        gfm_layer = self.gfm_encoder(sequence_emb, seq_mask, filter,output_all_encoded_layers=True)
        gfm_output = gfm_layer[-1]
        gfm_output = self.gather_indexes(gfm_output, item_seq_len-1)

        # LFM
        item_encoded_layers, total_lb_loss = self.lfm_encoder(sequence_emb, seq_mask, filter, output_all_encoded_layers=True)
        lfm_output = item_encoded_layers[-1]
        lfm_output = self.gather_indexes(lfm_output, item_seq_len - 1)
        
        concate_output = torch.cat((lfm_output, gfm_output),dim=-1)
        output = self.concat_layer(concate_output)

        last_hidden_state = self.gather_indexes(sequence_emb, item_seq_len - 1)
        output = self.LayerNorm(output + last_hidden_state)
        output = self.dropout(output)
        return output, gfm_output, lfm_output, total_lb_loss

     
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output, gfm_output, lfm_output, total_lb_loss = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]   
        
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        
        # add auxiliary loss
        logits = torch.matmul(gfm_output, test_item_emb.transpose(0,1))
        gfm_loss = self.loss_fct(logits, pos_items)
        logits = torch.matmul(lfm_output, test_item_emb.transpose(0,1))
        lfm_loss = self.loss_fct(logits, pos_items) 
        loss = loss + self.alpha*(gfm_loss + lfm_loss)

        # add load balancing loss
        loss += self.beta * total_lb_loss
        return loss
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output,_ ,_,_ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embeddings.weight
        
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) 
        return scores
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output,_ ,_,_  = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embeddings(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


   
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, 'silu':F.silu}


class LFMGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_bands = config['num_bands']
        
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),  
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.num_bands)
        )

    def forward(self, x):
        magnitude = x.abs()
        phase = torch.angle(x)
        mag_features = torch.mean(magnitude, dim=1)
        phase_features = torch.mean(phase, dim=1)
        combined_features = torch.cat([mag_features, phase_features], dim=-1)
        
        gate_logits = self.gate(combined_features)
        probs = F.softmax(gate_logits, dim=-1)
        
        local_band_prob, prob_indices = torch.topk(probs, self.num_bands, dim=-1)
        local_band_prob_normalized = local_band_prob / local_band_prob.sum(dim=-1, keepdim=True)
        
        return local_band_prob_normalized, prob_indices
    
    
class LFMfilterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.complex_weight1 = nn.Parameter(torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH']//2 + 1, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.conv_layers = config['conv_layers']
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.num_bands = config['num_bands']
        self.LFMgate = LFMGate(config)
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )
        self.LFMgate = LFMGate(config)

    def compute_balance_loss(self, local_band_indices, local_band_prob):
        batch_size = local_band_indices.size(0)
        mask = F.one_hot(local_band_indices, num_classes=self.num_bands).float()
        weighted_mask = mask * local_band_prob.unsqueeze(-1)
        band_usage = weighted_mask.sum(dim=[0, 1])
        band_usage = band_usage / batch_size
        ideal_usage = torch.ones_like(band_usage) * (1 / self.num_bands)
        usage_penalty = (band_usage - ideal_usage) ** 2
        balance_loss =  usage_penalty.mean()
        return balance_loss, band_usage
    
    def forward(self, input_tensor, seq_mask,filter):
        batch, max_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        local_band_prob,  prob_indices = self.LFMgate(x)
        balance_loss, band_usage = self.compute_balance_loss(prob_indices, local_band_prob)
        weight = torch.view_as_complex(self.complex_weight1)
        
        filtered_weight = torch.complex(filter * weight.real , filter* weight.imag)
        x_ = x * filtered_weight.permute(0,2,1)

        frequency_bands = torch.empty((batch, self.num_bands, max_len, hidden), device=input_tensor.device, dtype=input_tensor.dtype)
        for band in range(self.num_bands):
            frequency_output = torch.zeros_like(x_)
            band_start = band * (max_len//2+1) // self.num_bands
            band_end = (band + 1) * (max_len//2+1) // self.num_bands
            frequency_output[:,band_start:band_end] = x_[:,band_start:band_end]  
            sequence_emb_fft = torch.fft.irfft(frequency_output, n=max_len, dim=1, norm='ortho')
            
            band_output = self.out_dropout(sequence_emb_fft)
            frequency_bands[:,band] = self.LayerNorm(band_output + input_tensor)
            
        selected = torch.gather(frequency_bands, dim=1, index=prob_indices.view(batch, self.num_bands, 1, 1).expand(-1, -1, max_len, hidden))
        weighted_bands = local_band_prob.view(batch, self.num_bands, 1, 1) * selected
        LFM_output = weighted_bands.sum(dim=1)
        
        return LFM_output, balance_loss
    

class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(config['hidden_size'], config['inner_size'])
        if isinstance(config['hidden_act'], str):
            self.intermediate_act_fn = ACT2FN[config['hidden_act']]
        else:
            self.intermediate_act_fn = config['hidden_act']

        self.dense_2 = nn.Linear(config['inner_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LFMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.filterlayer = LFMfilterLayer(config)
        self.intermediate = Intermediate(config)
    def forward(self, hidden_states, seq_mask, filter):
        LFM_output, balance_loss = self.filterlayer(hidden_states, seq_mask, filter)
        output = self.intermediate(LFM_output)
        return output, balance_loss


class LFMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = LFMLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config['n_layers'])])

    def forward(self, hidden_states, seq_mask, filter ,output_all_encoded_layers=True):
        all_encoder_layers = []
        total_balance_loss = 0
        
        for layer_module in self.layer:
            hidden_states, balance_loss= layer_module(hidden_states, seq_mask, filter)
            total_balance_loss += balance_loss
            
            if output_all_encoded_layers:
                all_encoder_layers.append((hidden_states))
                
        if not output_all_encoded_layers:
            all_encoder_layers.append((hidden_states))
            
        return all_encoder_layers, total_balance_loss
    
    
class GFMFilterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.complex_weight1 = nn.Parameter(torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH']//2 + 1, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.conv_layers = config['conv_layers']
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.num_bands = config['num_bands']
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )


    def forward(self, input_tensor, seq_mask, filter):
        batch, max_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight1)
            
            
        filtered_weight = torch.complex(filter * weight.real , filter* weight.imag )
        x_ = x * filtered_weight.permute(0,2,1)
        
        whole_sequence_emb_irfft = torch.fft.irfft(x_, n=max_len, dim=1, norm='ortho')
        whole_emb = self.out_dropout(whole_sequence_emb_irfft)
        whole_emb = self.LayerNorm(whole_emb + input_tensor)
        return whole_emb


class GFMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.filterlayer = GFMFilterLayer(config)
        self.intermediate = Intermediate(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
            
    def forward(self, hidden_states, seq_mask, filter):
        gfm_output = self.filterlayer(hidden_states, seq_mask, filter)
        output = self.intermediate(gfm_output)
        return output


class GFMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = GFMLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config['n_layers'])])

    def forward(self, hidden_states, seq_mask, filter, output_all_encoded_layers=True):
        all_encoder_layers = []        
        for layer_module in self.layer:
            hidden_states= layer_module(hidden_states, seq_mask, filter)            
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers
