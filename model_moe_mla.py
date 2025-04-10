"""
GPT Language Model with DeepSeekMOE and MLA (Mixture of Layer Attention) support.
This implementation extends the base GPT model with:
1. DeepSeekMOE: Mixture of Experts for better parameter efficiency
2. MLA: Mixture of Layer Attention for improved attention mechanism
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Expert(nn.Module):
    """Single expert in the MOE system"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    """Mixture of Experts layer"""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.temperature = config.moe_temperature

    def forward(self, x):
        # Calculate routing weights
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits / self.temperature, dim=-1)
        
        # Get top-k experts
        top_k = min(2, self.num_experts)  # Use top-2 experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Compute weighted sum of expert outputs
        for i in range(top_k):
            expert_idx = top_k_indices[..., i]
            expert_probs = top_k_probs[..., i]
            
            # Get expert outputs
            expert_outputs = torch.stack([
                self.experts[idx](x[j]) for j, idx in enumerate(expert_idx)
            ])
            
            # Add weighted expert outputs
            output += expert_outputs * expert_probs.unsqueeze(-1)
        
        return output

class MLAAttention(nn.Module):
    """Multi-head Latent Attention with low-rank key-value joint compression and decoupled RoPE"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        
        # KV compression dimensions
        self.dc = 4 * self.head_size  # KV compression dimension
        self.dh_r = self.head_size // 2  # Decoupled RoPE dimension per head
        
        # Query compression
        self.W_DQ = nn.Linear(config.n_embd, self.dc * config.n_head, bias=config.bias)
        self.W_UQ = nn.Linear(self.dc * config.n_head, config.n_embd, bias=config.bias)
        
        # KV joint compression
        self.W_DKV = nn.Linear(config.n_embd, self.dc, bias=config.bias)
        self.W_UK = nn.Linear(self.dc, config.n_embd, bias=config.bias)
        self.W_UV = nn.Linear(self.dc, config.n_embd, bias=config.bias)
        
        # Decoupled RoPE components
        self.W_QR = nn.Linear(self.dc * config.n_head, self.dh_r * config.n_head, bias=config.bias)
        self.W_KR = nn.Linear(config.n_embd, self.dh_r, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
        # Initialize RoPE
        self.rope = RotaryPositionalEmbedding(self.head_size)

    def forward(self, x, use_cache=False, past_kv=None):
        B, T, C = x.size()
        
        # 1. Query compression
        c_Q = self.W_DQ(x)  # [B, T, dc * n_head]
        q_C = self.W_UQ(c_Q)  # [B, T, n_embd]
        
        # 2. KV joint compression
        c_KV = self.W_DKV(x)  # [B, T, dc]
        k_C = self.W_UK(c_KV)  # [B, T, n_embd]
        v_C = self.W_UV(c_KV)  # [B, T, n_embd]
        
        # 3. Decoupled RoPE components
        q_R = self.W_QR(c_Q)  # [B, T, dh_r * n_head]
        k_R = self.W_KR(x)    # [B, T, dh_r]
        
        # Apply RoPE
        q_R = self.rope(q_R)
        k_R = self.rope(k_R)
        
        # 4. Combine compressed and RoPE components
        q = torch.cat([q_C, q_R], dim=-1)  # [B, T, n_embd + dh_r * n_head]
        k = torch.cat([k_C, k_R], dim=-1)  # [B, T, n_embd + dh_r]
        
        # 5. Reshape for attention
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)  # [B, n_head, T, head_size]
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)  # [B, n_head, T, head_size]
        v_C = v_C.view(B, T, self.n_head, -1).transpose(1, 2)  # [B, n_head, T, head_size]
        
        # 6. Compute attention
        if use_cache and past_kv is not None:
            # Use cached KV
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v_C = torch.cat([past_v, v_C], dim=2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v_C
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 7. Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        if use_cache:
            # Return current KV for caching
            return y, (k, v_C)
        return y

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, ...] * x + self.sin_cached[:, :, :seq_len, ...] * x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MLAAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MoE(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    num_experts: int = 8  # Number of experts for MOE
    moe_temperature: float = 0.1  # Temperature for expert routing

class DeepSeekV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx 