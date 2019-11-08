import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import torch.nn.functional as F

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
        return self.decoder(trg, hn, enc_outputs = enc_out)[0] # [1] == hn

    def inference(self, src, beamsize=1):
        enc_output, hn = self.encoder(src)
        greedyresults = self.decoder.greedy(hn, enc_outputs = enc_output)
        beamresults =  self.decoder.beamsearch(hn, enc_outputs = enc_output, beamsize=self.args.beamsize)
        set_trace()
        print(beamresults);print(greedyresults)
        #''
        res = {'greedy': greedyresults, 'beam': beamresults}
        return res


##Decoder baseclass
class Decoder(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args

    def greedy(self, hidden, enc_outputs=None, maxlen=100):
        if enc_outputs is not None:
            bsz, srclen, hid2 = list(enc_outputs.shape)
            hid = hid2//2
            maxlen = int(srclen*1.5)
        else:
            _, bsz, hid = list(hidden.shape)

        output = SOS_TOKEN*torch.ones(bsz).long().to(self.args.device).unsqueeze(1)
        genidxs = PAD_TOKEN*torch.ones(bsz, maxlen).long().to(self.args.device)
        for t in range(maxlen):
            #self.rnn == decoder's rnn
            output, hidden = self.forward(output, hidden, enc_outputs=enc_outputs)
            output = output.squeeze().argmax(dim=1)
            genidxs[:, t] = output
            output= output.unsqueeze(1) # bsz, 1
            if (genidxs[:, t]==PAD_TOKEN).all():
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
            maxlen = int(1.5*srclen)

        ##make hidden batch_first => later seq first
        #hidden= hidden.transpose(1,0)
        _, bsz, hidsz = list(hidden.shape)

        hidden = hidden.repeat(1, beamsize, 1)

        beams = [Beam(beamsize, cuda=True) for i in range(bsz)]

        batch_idx = list(range(bsz))
        remainingsents = bsz

        for i in range(maxlen):
            ins = torch.stack([b.get_current_state() for b in beams if not b.done]).view(-1, 1) # remaining, beam  --> remaining*beam , 1
            out, hidden = self.forward(ins, hidden, enc_outputs=enc_outputs) #remaining*beam, 1, hid / 1, remaining*beam, hid

            wordlk=out.view(remainingsents, beamsize, -1).contiguous()
            active = []
            for b in range(bsz):
                if beams[b].done:
                    continue
                idx = batch_idx[b]
                if not beams[b].advance(wordlk.clone()[idx]):
                    active.append(b)

                _hidden = hidden.view(-1 ,beamsize, remainingsents, hidsz)[:,:,idx]

                ###below will update hidden and _hidden (chgs in _hidden will appear on hidden too)
                _hidden.data.copy_(
                    _hidden.index_select(1, beams[b].get_current_origin() )
                )
            if not active:
                break

            def update_active(t):
                # select only the remaining active sentences
                hsz = t.shape[-1]
                view = t.view(#t.data.view(
                    -1, remainingsents, hsz #model.decoder.hidden_size
                )

                new_size = list(t.shape)
                new_size[-2] = new_size[-2] * len(active_idx) // remainingsents

                return view.index_select(1, active_idx).view(*new_size)

            hidden = update_active(hidden) #[remaining, beam, hid]
            #trg_h = update_active(trg_h) # [1, beam*bsz, hid]
            #if len(active_idx) < batchsize:
            #    set_trace()
            enc_outputs = update_active(enc_outputs)

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
        self.emb = nn.Embedding(vocabsize, hid)
        self.rnn = nn.GRU(hid, hid, num_layers=1, batch_first=True)

    def forward(self, src):
        x = self.emb(src)
        batchsize = src.shape[0]
        hidden_size = x.shape[-1]
        h0 = torch.zeros(1, batchsize, hidden_size).to(self.args.device) # bidrectional 2
        enc_outputs, hn = self.rnn(x, h0)
        return None, hn

class RNNDec(Decoder):
    def __init__(self, args, vocabsize, hid, emb, **kwargs):
        super().__init__(args)
        self.args = args
        self.emb = emb #from Enc
        self.to_onehot = nn.Linear(hid, vocabsize)
        self.rnn = nn.GRU(hid, hid, num_layers=1, batch_first=True)

    def forward(self, trg, hn, enc_outputs=None):
        x = self.emb(trg) # bsz, 1, hid
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
        '''
        the directions can be separated using
        >>> output.view(seq_len, batch, num_directions, hidden_size),
        with forward and backward being direction 0 and 1 respectively.
        '''

    def forward(self, src):
        x = self.emb(src)
        batchsize = src.shape[0]
        hidden_size = x.shape[-1]
        h0 = torch.zeros(2, batchsize, hidden_size).to(self.args.device) # bidrectional 2
        return self.rnn(x, h0)

class RNNSearchDec(Decoder):
    def __init__(self, args, vocabsize, hid, emb, **kwargs):
        super().__init__(args)
        self.args = args
        self.emb = emb # from encoder
        self.to_onehot = nn.Linear(hid, vocabsize)
        self.rnn = nn.GRU(hid*2, hid, batch_first =True)
        self.attention = AddAttn(hid, hid)

    def forward(self, trg, h, enc_outputs=None):
        bsz, srclen, hid = list(enc_outputs.size())
        hid = hid//2

        attn = self.attention(h, enc_outputs)
        context = enc_outputs * attn.unsqueeze(2) # bsz srclen hid

        context = context.sum(dim=1) #bsz 1 hid
        x = self.emb(trg) # bsz, trglen, hid
        rnnin = torch.cat((x, context), dim=2) # bsz, trglen, hid

        h0 = torch.zeros(1, bsz, hid).to(self.args.device)
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
