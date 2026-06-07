import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding from scratch.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_len, d_model) filled with zeros
        pe = torch.zeros(max_len, d_model)
        
        # Create a vector of shape (max_len, 1) containing positions [0, 1, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term: 10000^(2i/d_model)
        # We work in log space for numerical stability: exp(2i * -log(10000)/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        # Add a batch dimension: shape becomes (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register pe as a buffer so it's saved in the state_dict but not updated by optimizer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        # Add the positional encoding up to the sequence length of x
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention from scratch.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Query, Key, and Value projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection layer
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        """
        q, k, v shapes: (batch_size, seq_len, d_model)
        """
        batch_size = q.size(0)
        
        # 1. Project Q, K, V and split into multiple heads
        # Shape change: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Compute Scaled Dot-Product Attention
        # Scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Mask out padded tokens (replace with a very large negative value)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax over the last dimension to get attention weights (probabilities)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 3. Multiply attention weights by V
        # Shape: (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, V)
        
        # 4. Concatenate heads and project back to d_model
        # Shape: (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)
        
        return output, attn_weights


class BaselineModel(nn.Module):
    """
    Baseline Architecture: Input -> Embedding -> Dense -> Output
    We use Global Average Pooling after the Embedding layer to get a fixed size representation.
    """
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # shape: (batch_size, seq_len, embedding_dim)
        
        # Global Average Pooling (average along the sequence dimension)
        pooled = embedded.mean(dim=1)  # shape: (batch_size, embedding_dim)
        
        logits = self.fc(pooled)  # shape: (batch_size, num_classes)
        return logits


class AttentionModel(nn.Module):
    """
    Self-Attention Architecture: Input -> Embedding -> MultiHeadAttention -> Dense -> Output
    Includes sinusoidal positional encoding added to the embeddings before self-attention.
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, num_classes, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len)
        # Create mask if padding is to be ignored: shape (batch_size, 1, 1, seq_len)
        if mask is None:
            # We construct a mask based on the padding index 0 in x
            # shape: (batch_size, 1, 1, seq_len)
            mask = (x != 0).unsqueeze(1).unsqueeze(2)
            
        # 1. Embeddings
        embedded = self.embedding(x)  # shape: (batch_size, seq_len, embedding_dim)
        
        # 2. Add Positional Encodings
        pe_embedded = self.pos_encoder(embedded)  # shape: (batch_size, seq_len, embedding_dim)
        
        # 3. Multi-Head Self-Attention
        attn_out, attn_weights = self.attention(pe_embedded, pe_embedded, pe_embedded, mask=mask)
        
        # 4. Mean Pooling
        pooled = attn_out.mean(dim=1)  # shape: (batch_size, embedding_dim)
        
        # 5. Classification
        logits = self.fc(pooled)  # shape: (batch_size, num_classes)
        
        return logits, attn_weights
