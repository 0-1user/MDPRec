"""MEANTIME mixed temporal attention adapted to the MDPRec next-item protocol."""
import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType
from recbole_baseline.meantime_layers import TransformerMeantimeBlock, MixedAttention, GELU
from types import SimpleNamespace


class MEANTIME(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.time_seq = config['TIME_FIELD'] + config['LIST_SUFFIX']
        self.hidden = config['hidden_size']
        self.abs_types = config['absolute_kernel_types'].split('-') if config['absolute_kernel_types'] else []
        self.rel_types = config['relative_kernel_types'].split('-') if config['relative_kernel_types'] else []
        heads = len(self.abs_types)+len(self.rel_types)
        if not heads or self.hidden % heads or self.hidden % 2:
            raise ValueError('MEANTIME hidden_size must be even and divisible by the number of temporal kernels')
        if any(k not in ('p','d','c') for k in self.abs_types) or any(k not in ('s','e','l') for k in self.rel_types):
            raise ValueError('Absolute kernels: p/d/c; relative kernels: s/e/l')
        self.time_unit = float(config['time_unit_divide'])
        self.freq = float(config['freq'])
        if self.time_unit <= 0 or self.freq <= 0:
            raise ValueError('time_unit_divide and freq must be positive')
        self.mask_id = self.n_items
        self.item_embedding = nn.Embedding(self.n_items+1, self.hidden, padding_idx=0)
        # Keep all MDPRec history items and reserve an additional query position.
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden)
        self.constant_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden))
        raw = dataset.inter_feat[config['TIME_FIELD']].double().reshape(-1)
        if self.time_seq in dataset.inter_feat:
            raw = torch.cat((raw, dataset.inter_feat[self.time_seq].double().reshape(-1)))
        if not torch.isfinite(raw).all():
            raise ValueError('MEANTIME requires finite Unix timestamps')
        nonzero = raw[raw != 0]
        if not nonzero.numel():
            raise ValueError('MEANTIME requires nonzero Unix timestamps')
        divisor = 1000. if nonzero.abs().median().item() >= 1e11 else 1.
        days = torch.floor(nonzero/divisor/86400).long()
        self.register_buffer('timestamp_divisor', torch.tensor(divisor, dtype=torch.double))
        self.register_buffer('first_day', days.min().cpu())
        self.day_embedding = nn.Embedding(int(days.max()-days.min())+2, self.hidden, padding_idx=0)
        dropout = config['dropout_prob']
        self.dropout = nn.Dropout(dropout)
        args = SimpleNamespace(hidden_units=self.hidden, dropout=dropout, residual_ln_type='pre')
        self.blocks = nn.ModuleList([TransformerMeantimeBlock(args, len(self.abs_types), len(self.rel_types))
                                    for _ in range(config['n_layers'])])
        self.head = nn.Sequential(nn.Linear(self.hidden, self.hidden), GELU(), nn.LayerNorm(self.hidden))
        self.bias = nn.Parameter(torch.zeros(self.n_items))
        self.loss_fct = nn.CrossEntropyLoss()
        self.initializer_range = config['initializer_range']
        self.apply(self._init_weights)
        nn.init.normal_(self.constant_embedding, std=self.initializer_range)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.day_embedding.weight[0].zero_()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, MixedAttention):
            nn.init.normal_(module.rel_position_bias, std=self.initializer_range)

    def _kernels(self, tokens, times, lengths):
        valid = tokens.ne(0)
        seconds = times.double()/self.timestamp_divisor
        positions = (torch.arange(tokens.size(1), device=tokens.device)[None]
                     + self.max_seq_length-lengths[:, None]).clamp(0, self.max_seq_length)
        day_ids = (seconds/86400).floor().long()-self.first_day+1
        known = valid & (day_ids > 0) & (day_ids < self.day_embedding.num_embeddings)
        day_ids = torch.where(known, day_ids, 0)
        absolute = []
        for kind in self.abs_types:
            if kind == 'p':
                embedding = self.position_embedding(positions)
            elif kind == 'd':
                embedding = self.day_embedding(day_ids)
            else:
                embedding = self.constant_embedding.expand(tokens.size(0), tokens.size(1), -1)
            absolute.append(self.dropout(embedding))
        difference = ((seconds[:, :, None]-seconds[:, None, :])/self.time_unit).float()
        relative = []
        for kind in self.rel_types:
            step = 2 if kind == 's' else 1
            frequency = self.freq ** (-torch.arange(0, self.hidden, step, device=tokens.device).float()/self.hidden)
            angle = difference[..., None]*frequency
            if kind == 's':
                embedding = torch.cat((angle.sin(), angle.cos()), -1)
            elif kind == 'e':
                embedding = torch.exp(-angle.abs())
            else:
                embedding = torch.log1p(angle.abs())
            relative.append(self.dropout(embedding))
        return absolute, relative

    def _forward_batch(self, item_seq, lengths, times):
        # Discard padding after the actual prefix, then append a prediction MASK.
        # Its timestamp is the last observed time, never the held-out target time.
        index = torch.arange(item_seq.size(1), device=item_seq.device)[None]
        valid = index < lengths[:, None]
        tokens = torch.cat((item_seq.masked_fill(~valid, 0), item_seq.new_zeros(item_seq.size(0), 1)), 1)
        timestamps = torch.cat((times.masked_fill(~valid, 0), times.new_zeros(times.size(0), 1)), 1)
        tokens.scatter_(1, lengths[:, None], self.mask_id)
        last_time = times.gather(1, (lengths-1)[:, None])
        timestamps.scatter_(1, lengths[:, None], last_time)
        mask = tokens.ne(0)[:, None, None, :]
        absolute, relative = self._kernels(tokens, timestamps, lengths)
        x = self.dropout(self.item_embedding(tokens))
        for layer, block in enumerate(self.blocks):
            x = block(x, mask, absolute, relative, layer, None)
        return self.head(self.gather_indexes(x, lengths))

    def forward(self, item_seq, item_seq_len, timestamps):
        # Relative kernels are B*T*T*H; bound peak evaluation memory without
        # overriding the shared eval_batch_size (4096).
        chunk = max(1, min(256, 2097152//((item_seq.size(1)+1)**2*self.hidden)))
        return torch.cat([self._forward_batch(item_seq[i:i+chunk], item_seq_len[i:i+chunk], timestamps[i:i+chunk])
                          for i in range(0, item_seq.size(0), chunk)], 0)

    def _representation(self, interaction):
        return self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN], interaction[self.time_seq])

    def calculate_loss(self, interaction):
        return self.loss_fct(self.full_sort_predict(interaction), interaction[self.POS_ITEM_ID])

    def predict(self, interaction):
        representation = self._representation(interaction)
        items = interaction[self.ITEM_ID]
        return (representation*self.item_embedding(items)).sum(-1)+self.bias[items]

    def full_sort_predict(self, interaction):
        return self._representation(interaction) @ self.item_embedding.weight[:self.n_items].t()+self.bias
