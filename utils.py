from time import time
from datetime import datetime
import torch
from contextlib import contextmanager



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
    print("time logged! @" + f"{args.log_path}/{get_dirname_from_args(args)}.time")
