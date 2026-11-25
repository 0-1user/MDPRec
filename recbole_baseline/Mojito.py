"""PyTorch port of the supplied Mojito TensorFlow model; see MOJITO_README.md."""
from datetime import datetime, timezone
import math
import torch
from torch import nn
from torch.nn import functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class CalendarEncoder(nn.Module):
    def __init__(self, dim, width, linspace, use_year, dropout):
        super().__init__()
        self.use_year = use_year
        count = 7 if use_year else 6
        self.period = nn.Parameter(torch.linspace(0, linspace, width).repeat(count, 1))
        self.coefficients = nn.Parameter(torch.empty(count, width, 6))
        nn.init.xavier_uniform_(self.coefficients)
        self.bias = nn.Parameter(torch.zeros(count, width))
        self.projection = nn.Linear(count * width, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, calendar):
        if not self.use_year:
            calendar = calendar[..., 1:]
        frequency = torch.pow(10., -self.period)[..., None] * torch.arange(1, 4, device=calendar.device)
        angle = calendar[..., None, None] * frequency
        encoded = (torch.cat((angle.sin(), angle.cos()), -1) * self.coefficients).sum(-1) + self.bias
        return self.dropout(self.projection(encoded.flatten(-2)))


class AdmixtureBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.item_norm = nn.LayerNorm(2 * dim, eps=1e-8)
        self.context_norm = nn.LayerNorm(dim, eps=1e-8)
        self.q_item, self.k_item = nn.Linear(2 * dim, dim), nn.Linear(2 * dim, dim)
        self.q_context, self.k_context = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.value = nn.Linear(2 * dim, heads * dim)
        self.mixture = nn.Sequential(nn.Linear(2, heads), nn.ReLU(), nn.Linear(heads, heads))
        self.output = nn.Identity() if heads == 2 else nn.Linear(heads * dim, 2 * dim)
        self.norm = nn.LayerNorm(2 * dim, eps=1e-8)
        self.ff = nn.Sequential(nn.Linear(2 * dim, heads * dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(heads * dim, 2 * dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, context, valid, sigma):
        batch, length, _ = seq.shape
        qi, qc = self.item_norm(seq), self.context_norm(context)
        scores = torch.stack((self.q_item(qi) @ self.k_item(seq).transpose(-1, -2),
                              self.q_context(qc) @ self.k_context(context).transpose(-1, -2)), -1)
        if self.training:
            scores = scores + torch.randn_like(scores) * sigma.square()
        scores = self.mixture(scores).permute(0, 3, 1, 2) / math.sqrt(self.dim)
        causal = torch.ones(length, length, device=seq.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        attention = self.dropout(scores.masked_fill(~allowed, -1e9).softmax(-1)) * valid[:, None, :, None]
        value = self.value(seq).view(batch, length, self.heads, self.dim).transpose(1, 2)
        seq = qi + self.output((attention @ value).transpose(1, 2).reshape(batch, length, -1))
        normalized = self.norm(seq)
        return (normalized + self.ff(normalized)) * valid[..., None]


class Mojito(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.time_field = config['TIME_FIELD']
        self.time_seq = self.time_field + config['LIST_SUFFIX']
        self.dim = config['hidden_size']
        self.lambda_glob = config['lambda_glob']
        self.lambda_trans = config['lambda_trans_seq']
        self.beta = config['fism_beta']
        self.n_fism = config['n_fism_items']
        self.loss_fct = nn.CrossEntropyLoss()
        if min(self.dim, self.n_fism, config['n_layers'], config['n_heads']) < 1:
            raise ValueError('Mojito dimensions, layers, heads and memory must be positive')
        if not 0 <= self.lambda_trans <= 1 or self.lambda_glob < 0 or self.beta <= 0:
            raise ValueError('Invalid Mojito mixture/FISM weights')
        self.item_embedding = nn.Embedding(self.n_items, self.dim, padding_idx=0)
        self.user_embedding = nn.Embedding(dataset.num(self.USER_ID), self.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.dim)
        self.calendar = CalendarEncoder(self.dim, config['tempo_embedding_dim'], config['tempo_linspace'],
                                        config['use_year'], config['dropout_prob'])
        self.blocks = nn.ModuleList([AdmixtureBlock(self.dim, config['n_heads'], config['dropout_prob'])
                                     for _ in range(config['n_layers'])])
        self.sigma = nn.Parameter(torch.full((2,), .1))
        self.norm = nn.LayerNorm(2 * self.dim, eps=1e-8)
        self.dropout = nn.Dropout(config['dropout_prob'])
        # Recover complete TRAINING histories from augmented prefixes and targets.
        # Neither validation nor test targets are passed to this constructor.
        histories = [set() for _ in range(dataset.num(self.USER_ID))]
        users = dataset.inter_feat[self.USER_ID].tolist()
        targets = dataset.inter_feat[self.ITEM_ID].tolist()
        first = set()
        for index, (uid, iid) in enumerate(zip(users, targets)):
            histories[uid].add(iid)
            if uid not in first:
                histories[uid].update(dataset.inter_feat[self.ITEM_SEQ][index].tolist())
                first.add(uid)
        flat, offsets = [], [0]
        evaluation_memory = []
        generator = torch.Generator().manual_seed(config['seed'])
        for history in histories:
            history.discard(0)
            ids = torch.tensor(sorted(history), dtype=torch.long)
            flat.extend(ids.tolist())
            offsets.append(len(flat))
            sample = ids[torch.randperm(len(ids), generator=generator)[:self.n_fism]]
            evaluation_memory.append(F.pad(sample, (0, self.n_fism-len(sample))))
        self.register_buffer('history_items', torch.tensor(flat, dtype=torch.long))
        self.register_buffer('history_offsets', torch.tensor(offsets, dtype=torch.long))
        self.register_buffer('evaluation_memory', torch.stack(evaluation_memory))
        # Cache calendar features per distinct timestamp, not per augmented prefix.
        times = torch.unique(torch.cat((dataset.inter_feat[self.time_field].reshape(-1).double(),
                                       dataset.inter_feat[self.time_seq].reshape(-1).double())))
        if not torch.isfinite(times).all():
            raise ValueError('Mojito timestamps must be finite Unix seconds or milliseconds')
        times = torch.unique(torch.cat((torch.zeros(1, dtype=torch.double), times.cpu())), sorted=True)
        nonzero_times = times[times != 0]
        if not nonzero_times.numel():
            raise ValueError('Mojito requires at least one nonzero Unix timestamp')
        # Contemporary Unix seconds have 10 digits; LastFM milliseconds have 13.
        # Keep raw timestamps as cache keys and convert only for calendar extraction.
        divisor = 1000.0 if nonzero_times.abs().median().item() >= 1e11 else 1.0
        self.register_buffer('timestamp_divisor', torch.tensor(divisor, dtype=torch.double))
        dates = [self._utc_datetime(t, divisor) for t in times]
        min_year = min(d.year for t, d in zip(times, dates) if t != 0)
        self.register_buffer('year_origin_buffer', torch.tensor(min_year))
        features = [[d.year-min_year+1, d.month, d.day, d.weekday()+1, d.timetuple().tm_yday,
                     d.isocalendar().week, d.hour] for d in dates]
        self.register_buffer('timestamps', times)
        features = torch.tensor(features, dtype=torch.float)
        features[times == 0] = 0
        self.register_buffer('calendar_features', features)
        for embedding in (self.item_embedding, self.user_embedding, self.position_embedding):
            nn.init.normal_(embedding.weight, std=.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.user_embedding.weight[0].zero_()

    @staticmethod
    def _utc_datetime(timestamp, divisor):
        try:
            return datetime.fromtimestamp(float(timestamp) / divisor, timezone.utc)
        except (ValueError, OverflowError, OSError) as error:
            raise ValueError(
                f'Mojito cannot parse timestamp {float(timestamp)} with divisor {divisor}; '
                'expected Unix seconds or milliseconds in a consistent unit'
            ) from error

    def _context(self, timestamps):
        indices = torch.searchsorted(self.timestamps, timestamps.contiguous().double()).clamp_max(len(self.timestamps)-1)
        # Validation/test timestamps may be absent from the training split.
        if torch.equal(self.timestamps[indices], timestamps.double()):
            features = self.calendar_features[indices]
        else:
            # Preserve the training year origin when encoding unseen query timestamps.
            unique, inverse = torch.unique(timestamps.detach().cpu().double(), return_inverse=True)
            origin = int(self.year_origin_buffer)
            divisor = float(self.timestamp_divisor)
            rows = []
            for t in unique:
                d = self._utc_datetime(t, divisor)
                rows.append([d.year-origin+1, d.month, d.day, d.weekday()+1, d.timetuple().tm_yday,
                             d.isocalendar().week, d.hour] if t != 0 else [0]*7)
            features = torch.tensor(rows, device=timestamps.device, dtype=torch.float)[inverse.to(timestamps.device)].reshape(*timestamps.shape, 7)
        return self.calendar(features)

    def forward(self, item_seq, item_seq_len, timestamps):
        valid = item_seq.ne(0)
        # Align positions with the left-padded original model.
        positions = (torch.arange(item_seq.size(1), device=item_seq.device)[None]
                     + self.max_seq_length-item_seq_len[:, None]).clamp(0, self.max_seq_length-1)
        position = self.position_embedding(positions)
        context = self._context(timestamps)
        seq = torch.cat((self.item_embedding(item_seq)*math.sqrt(self.dim)+position,
                         context*math.sqrt(self.dim)+position), -1)
        seq = self.dropout(seq)*valid[..., None]
        for block in self.blocks:
            seq = block(seq, context, valid, self.sigma)
            context = seq[..., self.dim:]
        return self.gather_indexes(self.norm(seq), item_seq_len-1)

    def _prepare(self, interaction):
        seq, length = interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]
        local = self.forward(seq, length, interaction[self.time_seq])
        # Same observable input as MDPRec: history timestamps only.
        query = interaction[self.time_seq].gather(1, (length-1)[:, None]).squeeze(1)
        query_context = self._context(query)
        # Sample the full training history, as in the supplied implementation.
        memory_ids = []
        if self.training:
            for uid in interaction[self.USER_ID].tolist():
                start, end = int(self.history_offsets[uid]), int(self.history_offsets[uid+1])
                ids = self.history_items[start:end]
                ids = ids[torch.randperm(len(ids), device=ids.device)[:self.n_fism]]
                memory_ids.append(F.pad(ids, (0, self.n_fism-len(ids))))
            ids = torch.stack(memory_ids)
        else:
            ids = self.evaluation_memory[interaction[self.USER_ID]]
        memory = torch.cat((self.user_embedding(interaction[self.USER_ID])[:, None], self.item_embedding(ids)), 1)
        mask = torch.cat((torch.ones_like(ids[:, :1], dtype=torch.bool), ids.ne(0)), 1)
        transformed = self._transform(self.item_embedding(seq), memory, mask)
        keep = seq.ne(0).clone()
        keep[:, 0] &= length < self.max_seq_length
        history = (transformed*keep[..., None]).sum(1)
        return local, query_context, memory, mask, history

    def _transform(self, items, memory, mask):
        logits = (items @ memory.transpose(-1, -2)).masked_fill(~mask[:, None], -torch.inf)
        weights = torch.exp(logits-self.beta*torch.logsumexp(logits, -1, keepdim=True))
        return items*(1-self.lambda_trans)+items*(weights @ memory)*self.lambda_trans

    def _scores(self, state, candidates):
        local, context, memory, mask, history = state
        items = self.item_embedding(candidates)
        if items.dim() == 2:
            items = items[None].expand(local.size(0), -1, -1)
        local_score = (local[:, None, :self.dim]*items).sum(-1)+(local[:, self.dim:]*context).sum(-1, keepdim=True)
        transformed = self._transform(items, memory, mask)
        transformed = (transformed+history[:, None])/self.max_seq_length
        global_score = (transformed*items).sum(-1)
        return local_score, global_score

    def calculate_loss(self, interaction):
        # Same objective as MDPRec: one next-item label over all item logits.
        scores = self._full_scores(self._prepare(interaction))
        return self.loss_fct(scores, interaction[self.POS_ITEM_ID])

    def predict(self, interaction):
        local, glob = self._scores(self._prepare(interaction), interaction[self.ITEM_ID][:, None])
        return (local+self.lambda_glob*glob).squeeze(1)

    def _full_scores(self, state):
        chunks = []
        # Bound temporary candidate/FISM tensors without changing common batch sizes.
        chunk_size = max(1, min(1024, 1048576 // (state[0].size(0) * max(self.n_fism+1, self.dim))))
        for start in range(0, self.n_items, chunk_size):
            ids = torch.arange(start, min(start+chunk_size, self.n_items), device=state[0].device)
            local, glob = self._scores(state, ids)
            chunks.append(local+self.lambda_glob*glob)
        return torch.cat(chunks, 1)

    def full_sort_predict(self, interaction):
        return self._full_scores(self._prepare(interaction))
