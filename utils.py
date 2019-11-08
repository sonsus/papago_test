from time import time
from datetime import datetime
import torch
from contextlib import contextmanager
from config import *

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
    return batch

def get_dirname_from_args(args):
    tb_logname = f"{args.model}_optim_{args.optimizer}_{args.learning_rate}_ep{args.max_epochs}"
    if args.lrschedule=='reducelronplateau':
        tb_logname += f"ROP{args.gamma}_{args.mode}_{args.threshold}"
    elif args.lrschedule=='noamscheduler':
        tb_logname +=f"NOAM{args.warmup}_{args.factor}_{args.hidden}"

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
            if vocab[j] not [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
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
    with (args.log_path / f"{get_dirname_from_args(args)}.time").open(mode = 'w') as f:
        f.write(f"spent {str(h)[:4]} hrs!\n")
        f.write(f"=spent {str(m)[:4]} mins!\n")
        f.write(f"=spent {str(s)[:4]} secs!\n")
    print("time logged! @" + f"{args.log_path}/time.txt")
