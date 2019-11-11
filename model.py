import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import torch.nn.functional as F
import random

from config import *
from beam_module import Beam
from utils import *

from pdb import set_trace

def get_model(args):
    if args.model in ['rnnsearch', 'seq2seq']:
        mdl = Seq2Seq(args)
    elif args.model in ['transformer']:
        mdl = Transformer(d_model=args.hidden_size) # default_settings
        print("layernorm, positional encoding needed")
        set_trace()
    return mdl.to(args.device)


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
        enc_out, hn = self.encoder(src)
        if self.args.model == 'seq2seq':
            return self.decoder(trg, hn, enc_outputs = enc_out)[0] # [1] == hn
        elif self.args.model == 'rnnsearch':
            bsz = src.shape[0]
            max_len = trg.shape[1]

            # tensor to store decoder outputs
            outputs = torch.zeros_like(src).unsqueeze(2).repeat(1,1,self.vocabsize).float()

            # encoder_outputs is all hidden states of the input sequence, back and forwards
            # hidden is the final forward and backward hidden states, passed through a linear layer
            encoder_outputs, hidden = self.encoder(src)

            # first input to the decoder is the <sos> tokens
            output = trg[:, 0] #bsz
            hidden = hidden.unsqueeze(0)

            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, enc_outputs= encoder_outputs)
                outputs[:, t] = output.squeeze()
                teacher_force = random.random() < self.args.teacher_forcing_ratio
                top1 = output.squeeze().argmax(dim=1)
                #outputs[t] = top1
                output = (trg[:, t] if teacher_force else top1) #top1 or trg[:, t] to be the next input

            return outputs


    def inference(self, src, beamsize=1):
        enc_output, hn = self.encoder(src)
        greedyresults = self.decoder.greedy(hn.squeeze(), enc_outputs = enc_output)
        beamresults =  self.decoder.beamsearch(hn, enc_outputs = enc_output, beamsize=self.args.beamsize)

        res = {'greedy': greedyresults, 'beam': beamresults}
        return res


##Decoder baseclass
class Decoder(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args


    def greedy(self, hidden, enc_outputs=None, maxlen=60):
        if enc_outputs is not None:
            bsz, srclen, hid2 = list(enc_outputs.shape)
            hid = hid2//2
            maxlen = int(srclen*1.5)
        else: # would not be visited
            _, bsz, hid = list(hidden.shape)


        output = SOS_TOKEN*torch.ones(bsz).long().to(self.args.device)
        genidxs = PAD_TOKEN*torch.ones(bsz, maxlen).long().to(self.args.device)

        p_trglen = expected_len_trg(srclen)
        for t in range(maxlen):
            #self.rnn == decoder's rnn
            output, hidden = self.forward(output, hidden, enc_outputs=enc_outputs)
            if t!=0 and t+1<p_trglen and self.args.heu_penalty<0:
                output = discourage_specials(output.squeeze(),
                                            dim=1,
                                            penalty=self.args.heu_penalty*(p_trglen-t+1)) #bsz

            output = output.squeeze().argmax(dim=1)
            genidxs[:, t] = output

            output= output.unsqueeze(1) # bsz, 1
            if (genidxs[:, t]==PAD_TOKEN).all() or (genidxs[:, t]==EOS_TOKEN).all():
                genidxs = genidxs[:,:t+1]
                break

        lens = (genidxs!=PAD_TOKEN).sum(dim=1)
        lens = [e.item() for e in lens]

        res = {
            'sentidxs': genidxs,
            'lens': len
        }
        return res

    #### modified from https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/decode.py
    def beamsearch(self, hidden, enc_outputs=None, maxlen=100, beamsize=4):
        if enc_outputs is not None:
            bsz, srclen, hidsz = enc_outputs.shape
            enc_outputs = enc_outputs.repeat(beamsize, 1, 1)
            #maxlen = int(1.5*srclen)

        hidden = hidden.unsqueeze(0)
        _, bsz, hidsz = list(hidden.shape)

        hidden = hidden.repeat(1, beamsize, 1)

        beams = [Beam(beamsize, cuda=True) for i in range(bsz)]

        batch_idx = list(range(bsz))
        remainingsents = bsz
        p_trglen = expected_len_trg(srclen)

        for i in range(p_trglen+10):
            ins = torch.stack([b.get_current_state() for b in beams if not b.done]).view(-1, 1) # remaining, beam  --> remaining*beam , 1
            out, hidden = self.forward(ins, hidden, enc_outputs=enc_outputs) #remaining*beam, 1, hid / 1, remaining*beam, hid

            if i!=0 and i+1<p_trglen and self.args.heu_penalty<0:
                out = discourage_specials(out.squeeze(),dim=1,penalty=self.args.heu_penalty*(p_trglen-i+1)) #bsz

            wordlk = out.view(remainingsents, beamsize, -1)

            active = []
            for b in range(bsz):
                if beams[b].done:
                    continue
                idx = batch_idx[b]
                if not beams[b].advance(wordlk.clone()[idx]):
                    active.append(b)

                _hidden = hidden.view(-1 ,beamsize, remainingsents, hidsz)[:,:,idx] #1 remain*beam hid => 1 beam reamin hid [:, :, idx] => 1 beam hid

                ###below will update hidden and _hidden (chgs in _hidden will appear on hidden too)
                _hidden.data.copy_(
                    _hidden.index_select(1, beams[b].get_current_origin() )
                )
            if not active:
                break

            active_idx =  [batch_idx[i] for i in active]
            batch_idx = {b: idx for idx, b in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                hid = t.shape[-1]
                view = t.view(#t.data.view(
                    -1, remainingsents, hid #model.decoder.hidden_size
                )

                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) // remainingsents
                aidx = torch.cuda.LongTensor(active_idx)
                return view.index_select(1, aidx).view(*new_size)

            def update_encout(t):
                hidsz_ = t.shape[-1]
                t=t.view(-1, remainingsents, hidsz_)
                aidx = torch.cuda.LongTensor(active_idx)
                return t.index_select(1, aidx).view(len(active)*beamsize, -1, hidsz_)

            hidden = update_active(hidden) #[remaining, beam, hid]
            #trg_h = update_active(trg_h) # [1, beam*bsz, hid]
            #if len(active_idx) < batchsize:
            #    set_trace()
            if self.args.model is 'rnnsearch':
                enc_outputs = update_encout(enc_outputs)

            remainingsents = len(active)

        #  (4) package everything up

        allhyps, allScores = [], []
        n_best = 1

        for b in range(bsz):
            scores, ks = beams[b].sort_best()

            allScores += [scores[:n_best]]
            hyps = [beams[b].get_hyp(k) for k in ks[:n_best]] #zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allhyps += [hyps]
        allhyps_, lenslist = beamsearch2tensorlist(self.args, allhyps)

        res = {
            'sentidxs': allhyps_,
            'scores': allScores,
            'lens': lenslist,
        }

        return res

##seq2seq
class RNNEnc(nn.Module):
    def __init__(self, args, vocabsize, hid, **kwargs):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.emb = nn.Embedding(vocabsize, hid)
        self.rnn = nn.GRU(hid, hid, num_layers=1, batch_first=True)

    def forward(self, src):
        x = self.dropout(self.emb(src))
        batchsize = src.shape[0]
        hidden_size = x.shape[-1]
        h0 = torch.zeros(1, batchsize, hidden_size).to(self.args.device) # bidrectional 2
        enc_outputs, hn = self.rnn(x, h0)
        return enc_outputs, hn

class RNNDec(Decoder):
    def __init__(self, args, vocabsize, hid, emb, **kwargs):
        super().__init__(args)
        self.args = args
        self.emb = emb #from Enc
        self.to_onehot = nn.Linear(hid, vocabsize)
        self.dropout = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(hid, hid, num_layers=1, batch_first=True)


    def forward(self, trg, hn, enc_outputs=None):
        #nothing happens if enc_outputs goes into here...
        x = self.dropout(self.emb(trg)) # bsz, 1, hid
        batchsize = trg.shape[0]
        hidden_size = hn.shape[-1]
        #h0 = torch.zeros(1, batchsize, hidden_size).to(args.device) # bidrectional 2
        hids, hn = self.rnn(x, hn)
        logits = self.to_onehot(hids)
        return logits, hn


##RNNSEARCH
##enchid == dechid
class RNNSearchEnc(nn.Module):
    def __init__(self, args, vocabsize, hid, **kwargs):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(vocabsize, hid)
        self.rnn = nn.GRU(hid, hid, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        '''
        the directions can be separated using
        >>> output.view(seq_len, batch, num_directions, hidden_size),
        with forward and backward being direction 0 and 1 respectively.
        '''
        self.fc = nn.Linear(2*hid, hid)

    def forward(self, src):
        x = self.dropout(self.emb(src))
        batchsize = src.shape[0]
        hidden_size = x.shape[-1]
        h0 = torch.zeros(2, batchsize, hidden_size).to(self.args.device) # bidrectional 2
        enc_out, hidden = self.rnn(x, h0)
        hidden = torch.tanh( self.fc( torch.cat( (hidden[-2], hidden[-1]), dim=1) ) ) # 256
        return enc_out, hidden

class RNNSearchDec(Decoder):
    def __init__(self, args, vocabsize, hid, emb, **kwargs):
        super().__init__(args)

        self.args = args
        self.emb = emb # tied embedding
        self.to_onehot = nn.Linear(hid, vocabsize)
        self.rnn = nn.GRU(hid*3, hid, batch_first =True)
        self.attention = AddAttn(args, 2*hid, hid)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, trg, h, enc_outputs=None): # trg: [bsz], h: [1, bsz, hid], enc_outputs: [bsz, srclen, 2hid]
        #set_trace()
        bsz, srclen, hid = list(enc_outputs.size())
        hid = hid//2

        attn = self.attention(h, enc_outputs)
        context = enc_outputs * attn.unsqueeze(2) # bsz srclen hid

        context = context.sum(dim=1).unsqueeze(1) #bsz 1 hid
        x = self.dropout(self.emb(trg)).unsqueeze(1) # bsz, 1, hid
        if x.dim()>3:
            x=x.squeeze(1)
        #set_trace()
        rnnin = torch.cat((x, context), dim=2) # bsz, , hid

        #h0 = torch.zeros(1, bsz, hid).to(self.args.device) # should be at seq2seq forward
        if h.dim()==2:
            h= h.unsqueeze(0)
        logits, hid = self.rnn(rnnin, h) # bsz, 1, hid \ 1, bsz, hid
        logits = self.to_onehot(logits) # bsz, 1, vocab
        return logits, hid

class AddAttn(nn.Module):
    def __init__(self, args, enc_hid, dec_hid):
        super().__init__()
        self.args = args
        self.enc_hid = enc_hid # 2hid
        self.dec_hid = dec_hid # hid

        self.attn = nn.Linear(self.enc_hid + self.dec_hid, self.dec_hid) # 3hid => 1hid
        self.v = nn.Linear(self.dec_hid, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        #set_trace()
        bsz, srclen, hid2 = list(enc_outputs.shape) #bsz srclen 2hid
        hid = hid2//2

        dec_hidden = dec_hidden.repeat(srclen, 1, 1).transpose(1,0)

        cat_tensor = torch.cat((dec_hidden, enc_outputs), dim=2) # bsz, srclen, 3hid
        energy = torch.tanh(self.attn(cat_tensor)) #bsz srclen hid
        attention = self.v(energy).squeeze(2)# bsz srclen

        return F.softmax(attention, dim=1)
