import torch
import torch.nn as nn
import math


# Input embeddings: they are universal, all the words in the vocabulary have a certain embedding corresponding to them representing the meaning of each word
class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model) # Mapping a vector embedding to every token in our vocabulary using dictionary -like things

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) # In the paper, the authors multliplied the embeddings with this quantity

# Positional encoding: Depends on the position of the word within a sequence/sentence
# Achieved by using the two distinct formulae of sin and cos for different positions in the sentence, this is modified using a logspace equation, which we are implementing below
class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # Each token in the sequence/sentence is assigned a seperate positional encoding info
        self.droupout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model)) # Starting from 0, going until end incrementing 2
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model)) # Starting from 2, going until end incrementing 2
        
        # Add a batch dimension to the positional encoding to represent several sequences in the dataset
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe) # Tensor pe is not just a parameter now, it is stored in the buffer along with the state of the model when we save

        def forward(self, x):
            x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (entire batch, seq_len of the respective x, d_model) # requires_grad_(False) as we don't want to learn this tensor
            return self.dropout(x)
        
# Layer normalization
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None: # epsilon is needed as seen in the normalization formula in the denominator, if the standard deviation is 0 or ~0, to avoid /0
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter that we multiply # nn.Parameter() makes a parameter learnable
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter that we add 

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1) 
        # calculating mean and std across the last dimension which is d_model or embedding size # In other words for each token
        # Usually the mean cancels the dimensions to which we apply, but, we want to keep it
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Feed forward 
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
# Multi-head attention
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq # dimesnsions of these matrices are (dmodel,dmodel), to make the o/p shape (seq,dmodel)
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout): # Defining as a static method, we can use this function without instantiating the corresponding class
        d_k = query.shape[-1] # last dimension of the Query, key or value
        
        # Just apply the formula from the paper

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len) # attention score matrices in each head
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # @ -> matrix multiplication, 
        #transpose(-2,-1) -> permuting the last two dimensions, actual key-> (batch,h,seq_len,dk), modified key-> (batch,h,dk,seq_len)

        # If mask is defined, apply it, else, normal softmax
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores # tuple is feeded forward in the model, returning attention scores to visualiza later

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) -> Q' in the paper
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) -> K' in the paper
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) -> V' in the paper

        # Dividing the Q', K' and V' into smaller matrices to give each of one to each head
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)-> using transpose from the previous step, we can do the same thing with permute
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # 1st dimension is same as we don't want to split based on batch, same with second dimension
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) # Need to split across 3rd dimension, embedding size, d_model, reshape using view
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) # View is used to get parts of tensors and modifications of the tensor created by view modify orig memory

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model) # contiguous() to keep the memory next to each other in pytorch
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer): # sublayer -> previous layer
        return x + self.dropout(sublayer(self.norm(x))) # We combine x with drouput(output of sublayer) in this context, sublayer in encoder is MH-A o/p
        # in paper, they have applied sublayer first and then applied normalization, however, in practice, it is found that first we do normalization and then apply sublayer


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask): # Mask we need to apply to the input of the encoder, we don't want to see the relation of padding words with other words, so we apply a mask across them
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # lambda x is the function we are applying first, self attention -> Q=K=V=X, src_mask
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# Encoder has many encoder blocks    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask) # Output of the previous encoder block is the input of the next block
        return self.norm(x)
    
# CODING A DECODER ------------------------------
# Word embeddings and positional encoder can be used as we have created in the encoder as they are the same
class DecoderBlock(nn.Module):
    ################################# We use self-attention in Masked multi-head attention and cross_attention later in Multi-Head Attention, So, we use both in decoder block
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # src_mask -> mask coming from encoder, tgt_mask -> mask coming from decoder = preventing the model to see future o/p words 
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask) # Just calling the forward method above
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model) 
        src = self.src_embed(src) # input embedding, positional encoding and encoder
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt) # target language's embedding, positional encoding, decoder
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x) # final linear layer

# function to initialize, instantiate everything n-> no of encoder blocks and decoder blocks,     
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout) # But, typically we don't have to crete two positional encoding layers as they are only dependent on sequence at hand
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) # Ofcourse the target vocabulary size as we want to convert the source language into target language
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters # Instead of starting with random values a lot of algorithms start from Xavier's uniform distribution
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer