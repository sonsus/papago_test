import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from config import *
from pdb import set_trace

class Transformers(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder=None, decoder=None, src_embed=None, tgt_embed=None, generator=None, vocab=None):
        super(Transformers, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.vocab = vocab

    @classmethod
    def resolve_args(cls, args, vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout_tr)
        position = PositionalEncoding(args.d_model, args.dropout_tr)
        model = Transformers(
            encoder = Encoder(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout_tr), args.N),
            decoder = Decoder(DecoderLayer(args.d_model, c(attn), c(attn),
                                 c(ff), args.dropout_tr), args.N),
            src_embed = nn.Sequential(Embeddings(args.d_model, len(vocab.itos)), c(position)),
            tgt_embed = nn.Sequential(Embeddings(args.d_model, len(vocab.itos)), c(position)),
            generator = Generator(args.d_model, len(vocab.itos)),
            vocab = vocab
            )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        #set_trace()

        return model # check model.generator / encoder / decoder / tgt_embed / src_embed

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        "Take in and process masked src and target sequences."
        #set_trace()
        return self.generator(self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def inference(self, src, merged_hs, src_mask=None, merged_mask=None, beamsize = 1):

        batch_size, num_choice, seqlen = merged_hs.shape

        merged_hs_flat = merged_hs.reshape(batch_size*num_choice, seqlen)
        merged_mask_flat = merged_mask.reshape(batch_size*num_choice, seqlen, seqlen)

        src_flat = src.repeat(num_choice, 1) # bsz, len -> bsz*numchoice, len

        probs = self.generator(self.decode( self.encode(src_flat, src_mask),
                                            src_mask,
                                            merged_hs_flat,
                                            merged_mask_flat
                                            )
                                ) # bsz*choice, seqlen, vocabsize

        prob_hs = probs.gather(index=merged_hs_flat.unsqueeze(-1), dim=2).squeeze() # bsz*choice, seqlen



        """calc PPL"""
        mask = (merged_hs_flat!= 0)&(merged_hs_flat!= 1)&(merged_hs_flat!=2) # BoolTensor[batch_size * (num_choices), max_len_h]
        len_each_hypothesis = (mask.sum(dim=1)).float()  # FloatTensor[batch_size * (num_choices)]
        prob_hs.masked_fill_(mask=(mask != 1), value=0.0)

        # ln ( PI(nth_rt(probs)) )
        logit = prob_hs.sum(dim=1) / len_each_hypothesis # [bsz*numchoice] (dim=1 tensor)
        ppl = torch.exp(-logit)

        """generate sentence"""

        genidxs = self.greedy_decode(src)

        results = {}
        results['PPL'] = ppl.reshape(batch_size, num_choice)
        results['trg_len'] = len_each_hypothesis.reshape(batch_size, num_choice)
        results['genidxs'] = genidxs

        return results

    def greedy_decode(self, src, src_mask=None):
        max_len = src.shape[-1]*2
        batch_size = src.shape[0]
        memory = self.encode(src, src_mask)
        genidxs = torch.ones(batch_size, 1).type_as(src)
        for i in range(max_len-1):
            out = self.decode(memory, src_mask, genidxs, subsequent_mask(genidxs.size(1)).type_as(src))
            prob = self.generator(out[:, -1])
            next_word = prob.argmax(dim = 1).unsqueeze(1)
            genidxs = torch.cat([genidxs, next_word], dim=1)

        return genidxs



class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocabl):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocabl)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
            ##set_trace()
        return self.norm(x) # check mask

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        #set_trace()
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            #set_trace()
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        #set_trace()
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    #set_trace()
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #set_trace()
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocabl):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocabl, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
