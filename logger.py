import os
import logging

from tqdm import tqdm
from tensorboardX import SummaryWriter
import tensorflow as tf
from typing import Dict, List
from utils import *


class Logger:
    def __init__(self, args):
        self.log_cmd = args.log_cmd
        log_name = get_dirname_from_args(args) + f"_{get_now()}"
        logdir = str(args.log_path / log_name)
        if str(args.nlg_path)==str(args.nlu_path) and str(args.nlg_path)=="DBG":
            logdir = 'log/DBG'
        args.log_path = logdir
        os.makedirs(logdir, exist_ok=True)
        self.tfboard = SummaryWriter(logdir)

    def __call__(self, name, val, n_iter):
        self.tfboard.add_scalar(name, val, n_iter)
        if self.log_cmd:
            tqdm.write(f'{n_iter}:({name},{val})')



def log_results(logger, name, state, step):
    for key, val in state.metrics.items():
        if isinstance(val, dict):
            for key2, v in val.items():
                logger(f"{name}/{key}/{key2}" , v, step)
        else:
            logger(f"{name}/{key}" , val, step)

def log_lr(logger, name, optimizer, ep):
    lr=0;
    for param_group in optimizer.param_groups:
        lr= param_group['lr']
        break

    logger(f"{name}", lr, ep)

def log_args(args):
    logdir = Path(args.log_path)
    #logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / "args.json").open(mode='w') as f:
        json.dump(str(args), f, indent=4 )
    print("args logged! @ " + f"{logdir}/args.json")


def get_logger(args):
    return Logger(args)
