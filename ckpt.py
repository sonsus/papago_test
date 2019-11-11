import os

import torch
from torch import nn
import torch.nn.functional as F

from dataset import load_data
from model import get_model
from utils import *
from optims import *

from pathlib import Path, PosixPath
from pprint import pprint as pp
import dill

from pdb import set_trace


def get_ckpt_path(args, epoch, loss=0.0000, cut_loss=False):
    ckpt_name = f"{args.model}"
    path = Path(args.ckpt_path) / ckpt_name / Path(args.log_path).name


    # make dir if not exists "mkdir -p"
    path.mkdir(exist_ok=True, parents=True)

    loss = '{:.4f}'.format(loss)
    ckptfile = path / f"ep.{epoch}_loss.{loss}.pth"

    if cut_loss:
        ckptfile = path / f"ep.{epoch}_loss*.pth"

    return ckptfile


def save_ckpt(args, model, vocab, epoch: int, loss: float, optimizer=None): # loss here is val loss
    print(f'saving epoch {epoch}')
    dt = {
        'args': args,
        'epoch': epoch,
        'loss': loss,
        'model': model.state_dict(),
        'vocab': vocab,
        'optimizer': optimizer.state_dict()
    }

    ckpt_path = get_ckpt_path(args, epoch, loss)
    print(f">>> Saving MODEL and VOCAB @ {ckpt_path} <<<")
    torch.save(dt, ckpt_path, pickle_module=dill)

def find_loss(ckpt_file):
    fname_only = Path(ckpt_file).name
    loss_str = fname_only.split('_loss.')[-1][:-4]
    loss_float = float(loss_str)
    return loss_float

#works great
def get_model_ckpt(args):

    fname_pattern = args.load_path+"/*"#get_ckpt_path(args, args.pretrained_ep, cut_loss=True)
    ckpt_paths = sorted(Path().glob(f'{fname_pattern}'), key=find_loss )# min loss loaded

    assert len(ckpt_paths) > 0, f"no ckpt candidate for {str(Path().glob(f'{fname_pattern}'))}"

    ckpt_path = ckpt_paths[0]  # monkey patch for choosing the best ckpt
    if load_path[-4:] == ".pth":
        ckpt_path = args.load_path
    print(f"\ncheckpoint loaded from {ckpt_path}")
    dt = torch.load(ckpt_path, pickle_module=dill)
    args.update(dt['args'])


    vocab = dt['vocab']

    its = load_data(args)
    model = get_model(args, vocab)
    model.load_state_dict(dt['model'])
    optimizer = get_optims(args, model)
    optimizer.load_state_dict(dt['optim_states'])

    dt['its'] = its
    dt['optimizer'] = optimizer
    dt['model'] = model

    return dt
