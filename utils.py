from time import time
from datetime import datetime
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from pathlib import Path

from config import *
from pdb import set_trace


def discourage_specials(tensor, dim=None, penalty=-100):#use this for augmenting decoder
    # length proportional penalty?
    # lower PAD, SOS, EOS prob
    t_penalty = torch.ones_like(tensor)*penalty
    out = tensor.unbind(dim=dim)[0] # to make bsz tensor
    discourageidx = torch.ones_like(out).long().unsqueeze(1) #bsz
    for i in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
        tensor.scatter_add_(dim, i*discourageidx, t_penalty)

    return tensor

def expected_len_trg(src_len):
    return int(0.551*src_len-0.406)


def word_drop(args, batch):
    if args.word_drop>0:
        bsz = batch.src.shape[0]
        len_s = batch.src.shape[1]
        len_t = batch.trg.shape[1]

        #0 is a normal token in this dataset.
        keepzero_s = 777*(batch.src==0).long()
        keepzero_t = 777*(batch.trg==0).long() #777 later be converted back to be 0's

        batch.src = batch.src.float()
        batch.trg = batch.trg.float()
        for i in range(bsz):

            s= (batch.src[i]!=PAD_TOKEN).sum().item()
            t= (batch.trg[i]!=PAD_TOKEN).sum().item()
            modif_rate_s = args.word_drop * (s/len_s)
            modif_rate_t = args.word_drop * (t/len_t)

            batch.src[i,:s] = F.dropout(batch.src[i,:s], p=modif_rate_s) * (1-modif_rate_s )
            batch.trg[i,:t] = F.dropout(batch.trg[i,:t], p=modif_rate_t) * (1-modif_rate_t )

        # dropped out parts should be ignored == PAD_TOKEN
        dropped_s = PAD_TOKEN*(batch.src==0).float()
        dropped_t = PAD_TOKEN*(batch.trg==0).float()
        batch.src = batch.src+dropped_s
        batch.trg = batch.trg+dropped_t

        # reconvert 777 to 0
        s777 = -777*(batch.src==777).float()
        t777 = -777*(batch.trg==777).float()
        batch.src = batch.src + s777
        batch.src = batch.trg + t777

        #float to long
        batch.src= batch.src.long()
        batch.trg= batch.trg.long()

    return batch

def make_fake_vocab():
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    l = len(alphabets)
    return [alphabets[i%l]*(1 + i//l ) for i in range(662)]

def get_now():
    now = datetime.now()
    return now.strftime('d%m%d_t%H%M')

def prep_batch(args, batch):
    batch.src.t_()
    batch.trg.t_()

    batch = word_drop(args, batch)

    return batch

def get_dirname_from_args(args):
    tb_logname = f"{args.model}_{args.optimizer}_ep{args.max_epochs}_labelsmooth{args.label_smoothing}"
    if args.lrschedule=='reducelronplateau':
        tb_logname += f"_{args.learning_rate}_ROP{args.gamma}_{args.mode}_{args.threshold}"
    elif args.lrschedule=='noamscheduler': #doesn't work for some reason..
        tb_logname +=f"NOAM{args.warmup}_{args.factor}_{args.hidden}"
    #elif args.lrschedule=='linear':
    #    tb_logname +=f"Lin_start{args.learning_rate}_end{args.endlr}_fromstep{args.start_step}"
    return tb_logname


def to_string(vocab, x): # greedy
    # x: float tensor of size BSZ X LEN X VOCAB_SIZE
    # or idx tensor of size BSZ X LEN
    if x.dim() == 3:
        x = x.argmax(dim=-1)

    res = []
    for i in range(x.shape[0]):
        sent = x[i]
        li = []
        for j in sent:
            if vocab[j] not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
                li.append(vocab[j])
            #if vocab.itos[j] not in vocab.specials:
                #li.append(vocab.itos[j])
        sent = ' '.join(li)
        res.append(sent)
    return res

def beamsearch2tensorlist(args, allhyps): # allhyps: [   [[]], [[]], [[]]  ]
    for i, hyps in enumerate(allhyps): # hyps: [[], [], []]
        lenslist = [len(h) for h in hyps]
        maxlen = max([len(h) for h in hyps]) # h: []
        st = [ h + [0]*(maxlen-len(h) ) for h in hyps ]
        allhyps[i] = torch.LongTensor(st).to(args.device)

    return allhyps, lenslist

def to_string2(vocab, x): # beamsearch x:list of tensors
    if isinstance(x, torch.Tensor):
        return to_string(vocab, x)
    elif x[0].dim()==2:
        res = []
        for batch_res in x:
            res.append(to_string(vocab, batch_res))
    elif x[0].dim()==1:
        x_= torch.stack(x)
        res = to_string(vocab, x_)
    else:
        """do not reach here"""
        set_trace()
    return res


@contextmanager
def log_time_n_args(args):
    start = time()
    yield
    s = time()-start
    m = s/60
    h = m/60
    with (Path(args.log_path) / f"{get_dirname_from_args(args)}.time").open(mode = 'w') as f:
        f.write(f"spent {str(h)[:4]} hrs!\n")
        f.write(f"=spent {str(m)[:4]} mins!\n")
        f.write(f"=spent {str(s)[:4]} secs!\n")
    print("time logged! @" + f"{args.log_path}/time.txt")
