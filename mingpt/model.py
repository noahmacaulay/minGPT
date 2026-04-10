"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn_type = getattr(config, "ffn_type", "mlp")
        self.last_aux_loss = None

        if self.ffn_type == "mlp":
            self.mlp = nn.ModuleDict(dict(
                c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
                act     = NewGELU(),
                dropout = nn.Dropout(config.resid_pdrop),
            ))
            self.ffn = None
        elif self.ffn_type == "moe":
            self.mlp = None
            self.ffn = MoELayer(
                n_embd=config.n_embd,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                expert_hidden_mult=config.moe_expert_hidden_mult,
                dropout=config.resid_pdrop,
            )
        elif self.ffn_type == "deepseek_moe":
            self.mlp = None
            self.ffn = DeepseekMoELayer(
                n_embd=config.n_embd,
                num_shared_experts=config.moe_num_shared_experts,
                num_routed_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                expert_hidden_mult=config.moe_expert_hidden_mult,
                dropout=config.resid_pdrop,
            )
        else:
            raise ValueError(f"Unknown ffn_type: {self.ffn_type}")

    def _forward_ffn(self, x):
        if self.ffn_type == "mlp":
            m = self.mlp
            self.last_aux_loss = None
            return m.dropout(m.c_proj(m.act(m.c_fc(x))))

        y, aux_loss, _ = self.ffn(x, return_aux_loss=True)
        self.last_aux_loss = aux_loss
        return y

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self._forward_ffn(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # feed-forward / MoE hyperparameters
        C.ffn_type = 'mlp'
        C.moe_num_experts = 4
        C.moe_top_k = 2
        C.moe_num_shared_experts = 1
        C.moe_expert_hidden_mult = 4
        C.moe_aux_loss_weight = 1e-2
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.moe_aux_loss_weight = config.moe_aux_loss_weight
        self.last_moe_aux_loss = None

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        moe_aux_loss = None
        for block in self.transformer.h:
            x = block(x)
            if block.last_aux_loss is not None:
                if moe_aux_loss is None:
                    moe_aux_loss = block.last_aux_loss
                else:
                    moe_aux_loss = moe_aux_loss + block.last_aux_loss
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        self.last_moe_aux_loss = moe_aux_loss

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if moe_aux_loss is not None:
                loss = loss + self.moe_aux_loss_weight * moe_aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

######################################
######### NOVEL CODE BELOW HERE ######
######################################


class FeedForwardExpert(nn.Module):
    """
    A small GPT-style feed-forward expert: Linear -> GELU -> Linear -> Dropout.
    """

    def __init__(self, n_embd, hidden_mult=4, dropout=0.0):
        super().__init__()
        hidden_size = hidden_mult * n_embd
        self.net = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, hidden_size),
            c_proj=nn.Linear(hidden_size, n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(dropout),
        ))

    def forward(self, x):
        m = self.net
        return m.dropout(m.c_proj(m.act(m.c_fc(x))))


class MoELayer(nn.Module):
    """
    A minimal top-k Mixture-of-Experts feed-forward layer.

    Each token is routed to `top_k` experts. The selected experts process only
    their assigned tokens, and their outputs are combined with the router
    probabilities.
    """

    def __init__(
        self,
        n_embd,
        num_experts,
        top_k=2,
        expert_hidden_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        assert num_experts > 0
        assert 1 <= top_k <= num_experts
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForwardExpert(
                n_embd=n_embd,
                hidden_mult=expert_hidden_mult,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

    def _load_balancing_loss(self, router_probs, topk_indices):
        tokens_per_expert = F.one_hot(topk_indices[:, 0], num_classes=self.num_experts).float().mean(dim=0)
        router_prob_per_expert = router_probs.mean(dim=0)
        return self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

    def forward(self, x, return_aux_loss=False):
        B, T, C = x.size()
        x_flat = x.view(B * T, C)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        y_flat = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            expert_mask = topk_indices == expert_id
            if not expert_mask.any():
                continue

            token_positions, topk_slots = expert_mask.nonzero(as_tuple=True)
            expert_inputs = x_flat[token_positions]
            expert_outputs = expert(expert_inputs)
            expert_weights = topk_probs[token_positions, topk_slots].unsqueeze(-1)
            y_flat.index_add_(0, token_positions, expert_outputs * expert_weights)

        y = y_flat.view(B, T, C)

        if not return_aux_loss:
            return y

        aux_loss = self._load_balancing_loss(router_probs, topk_indices)
        stats = {
            "router_logits": router_logits.view(B, T, self.num_experts),
            "router_probs": router_probs.view(B, T, self.num_experts),
            "topk_indices": topk_indices.view(B, T, self.top_k),
            "topk_probs": topk_probs.view(B, T, self.top_k),
        }
        return y, aux_loss, stats


class DeepseekMoELayer(nn.Module):
    """
    A small DeepSeek-MoE-style feed-forward layer.

    Shared experts are applied densely to every token. Routed experts are
    selected sparsely with top-k routing, then added on top of the shared path.
    """

    def __init__(
        self,
        n_embd,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2,
        expert_hidden_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        assert num_shared_experts >= 0
        assert num_routed_experts > 0
        assert 1 <= top_k <= num_routed_experts
        self.n_embd = n_embd
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        self.shared_experts = nn.ModuleList([
            FeedForwardExpert(
                n_embd=n_embd,
                hidden_mult=expert_hidden_mult,
                dropout=dropout,
            )
            for _ in range(num_shared_experts)
        ])
        self.router = nn.Linear(n_embd, num_routed_experts, bias=False)
        self.routed_experts = nn.ModuleList([
            FeedForwardExpert(
                n_embd=n_embd,
                hidden_mult=expert_hidden_mult,
                dropout=dropout,
            )
            for _ in range(num_routed_experts)
        ])

    def _shared_path(self, x_flat):
        if len(self.shared_experts) == 0:
            return torch.zeros_like(x_flat)

        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)
        return shared_out / len(self.shared_experts)

    def _load_balancing_loss(self, router_probs, topk_indices):
        tokens_per_expert = F.one_hot(topk_indices[:, 0], num_classes=self.num_routed_experts).float().mean(dim=0)
        router_prob_per_expert = router_probs.mean(dim=0)
        return self.num_routed_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

    def forward(self, x, return_aux_loss=False):
        B, T, C = x.size()
        x_flat = x.view(B * T, C)

        shared_out = self._shared_path(x_flat)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.routed_experts):
            expert_mask = topk_indices == expert_id
            if not expert_mask.any():
                continue

            token_positions, topk_slots = expert_mask.nonzero(as_tuple=True)
            expert_inputs = x_flat[token_positions]
            expert_outputs = expert(expert_inputs)
            expert_weights = topk_probs[token_positions, topk_slots].unsqueeze(-1)
            routed_out.index_add_(0, token_positions, expert_outputs * expert_weights)

        y = (shared_out + routed_out).view(B, T, C)

        if not return_aux_loss:
            return y

        aux_loss = self._load_balancing_loss(router_probs, topk_indices)
        stats = {
            "router_logits": router_logits.view(B, T, self.num_routed_experts),
            "router_probs": router_probs.view(B, T, self.num_routed_experts),
            "topk_indices": topk_indices.view(B, T, self.top_k),
            "topk_probs": topk_probs.view(B, T, self.top_k),
        }
        return y, aux_loss, stats
