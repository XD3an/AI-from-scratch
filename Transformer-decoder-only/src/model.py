import math
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import json


# Huperparameters
with open('config.json', 'r') as f:
    config = json.load(f)
class ModelConfig:
    """Configuration for the model"""
    CONTEXT_LENGTH: int = config['model']['context_length']     # Number of context frames (default: 16)
    D_MODEL: int = config['model']['d_model']                   # Dimension of the model (must be divisible by NUM_HEADS)
    D_FF: int = config['model']['d_ff']                         # Dimension of the feed forward network (must be equal to D_MODEL*4)
    NUM_BLOCKS: int = config['model']['num_blocks']             # Number of transformer blocks in the model (default: 8)
    NUM_HEADS: int = config['model']['num_heads']               # Number of heads in multi-head attention (must divide D_MODEL)
    DROP_OUT: float = config['model']['dropout']                # Drop out rate for regularization (default: 0.1)
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(ModelConfig.D_MODEL, ModelConfig.D_FF),
            nn.ReLU(),
            nn.Linear(ModelConfig.D_FF, ModelConfig.D_MODEL),
            nn.Dropout(ModelConfig.DROP_OUT)
        )
    
    def forward(self, x):
        return self.ffn(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(ModelConfig.D_MODEL, ModelConfig.D_MODEL // ModelConfig.NUM_HEADS, bias=False)
        self.wk = nn.Linear(ModelConfig.D_MODEL, ModelConfig.D_MODEL // ModelConfig.NUM_HEADS, bias=False)
        self.wv = nn.Linear(ModelConfig.D_MODEL, ModelConfig.D_MODEL // ModelConfig.NUM_HEADS, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(ModelConfig.CONTEXT_LENGTH, ModelConfig.CONTEXT_LENGTH)))
        self.drop_out = nn.Dropout(ModelConfig.DROP_OUT)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape # B: batch size, T: number of tokens, C: dimension of the model
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Compute the scaled dot product attention
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(ModelConfig.D_MODEL // ModelConfig.NUM_HEADS)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.drop_out(weights)
        
        output = weights @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(ModelConfig.NUM_HEADS)])
        self.projection_layer = nn.Linear(ModelConfig.D_MODEL, ModelConfig.D_MODEL)
        self.drop_out = nn.Dropout(ModelConfig.DROP_OUT)

    def forward(self, x):
        # Apply the attention heads
        attention_heads = [head(x) for head in self.heads]
        attention_heads = torch.cat(attention_heads, dim=-1)
        
        # Apply the projection layer
        output = self.projection_layer(attention_heads)
        output = self.drop_out(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(ModelConfig.D_MODEL)
        self.layer_norm2 = nn.LayerNorm(ModelConfig.D_MODEL)
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward_network = FeedForwardNetwork()

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feed_forward_network(self.layer_norm2(x))
        return x

class Model(nn.Module):
    def __init__(self,  max_token_value=100256): # default tiktoken cl100k vocab size
        super(Model, self).__init__()
        self.embedding = nn.Embedding(max_token_value, ModelConfig.D_MODEL)
        self.transformer_blocks = nn.Sequential(*(
            [TransformerBlock() for _ in range(ModelConfig.NUM_BLOCKS)] +
            [nn.LayerNorm(ModelConfig.D_MODEL)]
        ))
        self.model_out_linear_layer = nn.Linear(ModelConfig.D_MODEL, max_token_value)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embed the input tokens and add positional encoding
        positional_encoding_matrix = torch.zeros((ModelConfig.CONTEXT_LENGTH, ModelConfig.D_MODEL)).to(ModelConfig.DEVICE)
        position = torch.arange(0, ModelConfig.CONTEXT_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, ModelConfig.D_MODEL, 2).float() * (-math.log(10000.0) / ModelConfig.D_MODEL))
        positional_encoding_matrix[:, 0::2] = torch.sin(position * div_term)
        positional_encoding_matrix[:, 1::2] = torch.cos(position * div_term)
        positional_embedding = positional_encoding_matrix[:T, :].to(ModelConfig.DEVICE)
        x = self.embedding(idx) + positional_embedding

        # Pass the input through the transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # get the final logits
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -min(ModelConfig.CONTEXT_LENGTH, idx.size(1)):]
            logits, _ = self.forward(idx_crop)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
