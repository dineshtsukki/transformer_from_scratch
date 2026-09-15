import torch
import torch.nn as nn
import math 
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention as described in the paper "Attention is All You Need".

    Args:
       d_model: int: Dimensionality of the input embeddings.
       num_head : int: Number of parallel attention heads. 
       dropout: Droput probability. Default is 0.1.
    """
    def __init__(self,d_model: int,num_heads:int,dropout:float=0.1):
        super(MultiHeadAttention,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads # Dimension per head

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Projection layers for Q, K, V (no bias in original paper)
        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)

        # Output projection
        self.W_o=nn.Linear(d_model,d_model)

        # Dropout for attention weights
        self.dropout=nn.Dropout(dropout)

    def forward(self,query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,mask: torch.Tensor=None):
        """
        Forward pass for multi-head attention.

        Args:
            query: torch.Tensor: Query tensor of shape (batch_size, seq_len, d_model).
            key: torch.Tensor: Key tensor of shape (batch_size, seq_len, d_model).
            value: torch.Tensor: Value tensor of shape (batch_size, seq_len, d_model).
            mask: torch.Tensor: Mask tensor of shape (batch_size, seq_len).
        """
        batch_size=query.size(0)

        Q=self.W_q(query) # (batch_size, seq_len, d_model)
        K=self.W_k(key) # (batch_size, seq_len, d_model)
        V=self.W_v(value) # (batch_size, seq_len, d_model)

        # ===== Step 2: Split into Multiple Heads =====
        # Reshape to (batch, num_heads, seq_len, d_k)
        Q=Q.view(batch_size,-1,self.num_heads,self.d_head).transpose(1,2)
        K=K.view(batch_size,-1,self.num_heads,self.d_head).transpose(1,2)
        V=V.view(batch_size,-1,self.num_heads,self.d_head).transpose(1,2)

        ## Calculating attention Scores 
        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_head)

        # Applying mask for decoder 

        if mask is not None:
            scores.masked_fill(mask==0,-1e9)

        # Softmax and droput
        attention_weights=F.softmax(scores,dim=-1)
        attention_weights=self.dropout(attention_weights)

        # Weighted sum of values 
        attn_output=torch.matmul(attention_weights,V)

        # Reshape back to (batch_size, seq_len, d_model)
        attn_output=attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        output=self.W_o(attn_output)
        return output
    
class positionalEncoding(torch.nn.Module):
    def __init__(self,d_model:int,max_len:int=5000):
        super(positionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=0.1)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1).float()

        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        self.register_buffer("pe",pe.unsqueeze(0))

    def forward(self,x:torch.Tensor):
        x=x+self.pe[:,:x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    """
    Implements the Feedforward Neural Network(MLP) having one hidden layer.
    """
    def __init__(self,d_model:int,dh:int=2048,p:float=0.1):
        super(FeedForward,self).__init__()
        self.linear1=nn.Linear(d_model,dh)
        self.linear2=nn.Linear(dh,d_model)
        self.dropout=nn.Dropout(p)
    
    def forward(self,x:torch.Tensor):
        x=F.relu(self.linear1(x))
        x=self.dropout(x)
        x=self.linear2(x)
        return x
    
class EncoderLayer(nn.Module):
    """
    Implements a Encoder Layer in the Transformer Architecture.
    """
    def __init__(self,d_model:int,num_heads:int,dh:int,dropout:float=0.1):
        super(EncoderLayer,self). __init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads,dropout)
        self.ffn=FeedForward(d_model,dh,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor,mask:torch.Tensor=None):
        # === Self Attention =====
        attn_output=self.self_attn(x,x,x,mask)
        x=self.norm1(x+self.dropout(attn_output))  # Residual connection and layer norm

        # === Feedforward Layer ====
        ffn_output=self.ffn(x)
        x=self.norm2(x+self.dropout(ffn_output)) # Residual connection and layer norm

        return x
    
class Encoder(nn.Module):
    """
    Implements the Encoder in the Transformer Architecture.A series of N identical encoder layers 
    """
    def __init__(self,d_model:int,num_heads:int,dh:int,n_layers:int,dropout:float=0.1):
        super(Encoder,self).__init__()
        self.layers=nn.ModuleList([
            EncoderLayer(d_model,num_heads,dh,dropout) for _ in range(n_layers)
        ])

    def forward(self,x:torch.Tensor,mask:torch.Tensor=None):
        for layer in self.layers:
            x=layer(x,mask)
        return x

class DecoderLayer(nn.Module):
    """
    Implements a single Decoder Layer in the Transformer Architecture.
    Includes:
    - Masked Multi-Head Attention
    - Cross-Attention
    - Feedforward Neural Network
    - Residual Connections and Layer Notmalization
    """
    def __init__(self,d_model:int, num_head:int, dh:int, dropout:float=0.1):
        super(DecoderLayer,self).__init__()
        # Masked Multi-Head Attention
        self.self_attn=MultiHeadAttention(d_model,num_head,dropout)

        # Cross-Attention
        self.cross_attn=MultiHeadAttention(d_model,num_head,dropout)

        # Feedforward Neural Network
        self.ffn=FeedForward(d_model,dh,dropout)

        # Layer Norms
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)

        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor,enc_output:torch.Tensor,tgt_mask:torch.Tensor=None,src_mask:torch.Tensor=None):
        # Masked Multi-Head Attention
        attn_output=self.self_attn(x,x,x,tgt_mask)
        x=self.norm1(x+self.dropout(attn_output))

        # Cross-Attention
        cross_attn_output=self.cross_attn(x,enc_output,enc_output,src_mask)
        x=self.norm2(x+self.dropout(cross_attn_output))

        # Feedforward Neural Network
        ffn_output=self.ffn(x)
        x=self.norm3(x+self.dropout(ffn_output))

        return x

class Decoder(nn.Module):
    """
    Stack multiple DecoderLayer instances to form the full Decoder.
    """
    def __init__(self,d_model:int,num_heads:int,dh:int,n_layers:int,dropout:float=0.1):
        super(Decoder,self).__init__()
        self.layers=nn.ModuleList([
            DecoderLayer(d_model,num_heads,dh,dropout) for _ in range(n_layers)
        ])

    def forward(self,x:torch.Tensor,enc_output:torch.Tensor,tgt_mask:torch.Tensor=None,src_mask:torch.Tensor=None):
        for layer in self.layers:
            x=layer(x,enc_output,tgt_mask,src_mask)
        return x
    
class Transformer(nn.Module):
    """
    A full Transformer model that ties together the Encoder and Decoder.
    """
    def __init__(self,src_vocab_size:int,tgt_vocab_size:int,d_model:int,n_heads:int,ff_dim:int,num_encoder_layers:int,num_decoder_layers:int,max_seq_len:int=512,dropout:float=0.1):
        super(Transformer,self).__init__()

        # Embeddings 
        self.src_embeddings=nn.Embedding(src_vocab_size,d_model)
        self.tgt_embeddings=nn.Embedding(tgt_vocab_size,d_model)

        # Positional Encodings
        self.positional_encoding=positionalEncoding(d_model,max_seq_len)

        # Encoder and Decoder
        self.encoder=Encoder(d_model,n_heads,ff_dim,num_encoder_layers,dropout)
        self.decoder=Decoder(d_model,n_heads,ff_dim,num_decoder_layers,dropout)

        # Final linear projection to vocab size 
        self.output_projection=nn.Linear(d_model,tgt_vocab_size)

        # Droput for embeddings 
        self.dropout=nn.Dropout(dropout)

    def forward(self,src:torch.Tensor,tgt:torch.Tensor,src_mask:torch.Tensor=None,tgt_mask:torch.Tensor=None):
        src_emb=self.dropout(self.src_embeddings(src))
        tgt_emb=self.dropout(self.tgt_embeddings(tgt))

        src_emb=self.positional_encoding(src_emb)
        tgt_emb=self.positional_encoding(tgt_emb)

        enc_output=self.encoder(src_emb,src_mask)
        dec_output=self.decoder(tgt_emb,enc_output,tgt_mask,src_mask)

        output=self.output_projection(dec_output)
        return output




