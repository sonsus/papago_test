import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import torch.nn.functional as F
from config import *

def get_model(args):
    if args.model in ['rnnsearch', 'seq2seq']:
        mdl = Seq2Seq(args)
    elif args.model in ['transformer']:
        mdl = Transformer(d_model=args.hidden_size) # default_settings
        print("layernorm, positional encoding needed")
        set_trace()
    return mdl


## when args.model in ['seq2seq', 'rnnsearch']
class Seq2Seq(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.vocabsize = EOS_TOKEN+1
        self.hidden_size = args.hidden_size
        emod = RNNEnc if self.args.model == "seq2seq" else RNNSearchEnc
        dmod = RNNDec if self.args.model == "seq2seq" else RNNSearchDec

        self.encoder = emod(self.args, self.vocabsize, self.hidden_size)
        emb = self.encoder.emb
        self.decoder = dmod(self.args, self.vocabsize, self.hidden_size, emb)

    def forward(self, src, trg):
        return self.decoder(trg, self.encoder(src))


##seq2seq
class RNNEnc(nn.Module):
    def __init__(self, args, vocabsize, hid, **kwargs):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(vocabsize, hid)
        self.rnn = nn.GRU(hid, hid, num_layers=1, batch_first=True)

    def forward(self, src):
        x = self.emb(src)
        batchsize = src.shape[0]
        hidden_size = x.shape[-1]
        h0 = torch.zeros(1, batchsize, hidden_size).to(args.device) # bidrectional 2
        enc_outputs, hn = self.rnn(x, h0)
        return hn

class RNNDec(nn.Module):
    def __init__(self, args, vocabsize, hid, emb, **kwargs):
        super().__init__()
        self.args = args
        self.emb = emb #from Enc
        self.to_onehot = nn.Linear(hid, vocabsize)
        self.rnn = nn.GRU(hid, hid, num_layers=1, batch_first=True)

    def forward(self, trg, hn):
        x = self.emb(trg)
        batchsize = trg.shape[0]
        hidden_size = hn.shape[-1]
        #h0 = torch.zeros(1, batchsize, hidden_size).to(args.device) # bidrectional 2
        hids, hn = self.rnn(x, hn)
        logits = self.to_onehot(hids)
        return logits


##RNNSEARCH
##enchid == dechid
class RNNSearchEnc(nn.Module):
    def __init__(self, args, vocabsize, hid, **kwargs):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(vocabsize, hid)
        self.rnn = nn.GRU(hid, hid, num_layers=1, bidirectional=True, batch_first=True)
        '''
        the directions can be separated using
        >>> output.view(seq_len, batch, num_directions, hidden_size),
        with forward and backward being direction 0 and 1 respectively.
        '''

    def forward(self, src):
        x = self.emb(src)
        batchsize = src.shape[0]
        hidden_size = x.shape[-1]
        h0 = torch.zeros(2, batchsize, hidden_size).to(args.device) # bidrectional 2
        return self.rnn(x, h0)

class RNNSearchDec(nn.Module):
    def __init__(self, args, vocabsize, hid, emb, **kwargs):
        super().__init__()
        self.args = args
        self.emb = emb # from encoder
        self.to_onehot = nn.Linear(hid, vocabsize)
        self.rnn = nn.GRU(hid*2, hid, batch_first =True)
        self.attention = AddAttn(hid, hid)

    def forward(self, trg, enc_outputs, h):
        bsz, srclen, hid = list(enc_outputs.size())
        hid = hid//2

        attn = self.attention(h, enc_outputs)
        context = enc_outputs * attn.unsqueeze(2) # bsz srclen hid

        context = context.sum(dim=1) #bsz 1 hid
        x = self.emb(trg) # bsz, trglen, hid
        rnnin = torch.cat((x, context), dim=2) # bsz, trglen, hid

        h0 = torch.zeros(1, bsz, hid).to(args.device)
        hids = self.rnn(rnnin, h0)
        logits = self.to_onehot(hids)
        return logits

class AddAttn(nn.Module):
    def __init__(self, args, enc_hid, dec_hid):
        super().__init__()
        self.args = args
        self.enc_hid = enc_hid # 2hid
        self.dec_hid = dec_hid # hid

        self.attn = nn.Linear(self.enc_hid + self.dec_hid, self.dec_hid) # 3hid => 1hid
        self.v = nn.Linear(self.dec_hid, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        bsz, srclen = enc_outputs.shape[2] #2hid

        dec_hidden = dec_hidden.repeat(1, srclen, 1) # [bsz, srclen, hid]

        cat_tensor = torch.cat((dec_hidden, enc_outputs), dim=2) # bsz, srclen, 3hid
        energy = torch.tanh(self.attn(cat_tensor)) #bsz srclen hid
        attention = self.v(energy).squeeze(2)# bsz srclen

        return F.softmax(attention, dim=1)
