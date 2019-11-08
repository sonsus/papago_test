from ignite.engine.engine import Engine, State, Events
from ignite.metrics import Loss

from dataset import load_data
from model import get_model
from lrscheduler import get_scheduler
from logger import *

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import *
from metrics import Ngram
from eval import get_eval, runeval

from pprint import pprint

from pdb import set_trace

def get_trainer(args, model_, loss_fn, optimizer):
    def update_model(trainer, batch):
        model_.train()
        args._training =  model_.training

        optimizer.zero_grad()
        batch = prep_batch(args, batch)
        if args.model in ["rnnsearch", "seq2seq"]:
            y_pred = model_(batch.src, batch.trg)
        elif args.model == 'transformer':
            #trg_mask
            print("masks needs to be implemented here")
        else: # do not reach here
            exit(f"shouldn\'t reach {args}")
        loss = loss_fn(y_pred, batch.trg)
        loss.backward()
        if args.model in ['seq2seq', 'rnnsearch']:
            nn.utils.clip_grad_norm_(model_.parameters(), 1) # clip gradient
        optimizer.step()

        return y_pred.detach(), batch.trg

    trainer = Engine(update_model)
    metrics = {
        'loss': Loss(loss_fn, output_transform= lambda x: (x[0], x[1]) ),
        'ngrams': Ngram(args, make_fake_vocab(), output_transform=lambda x:(x[0], x[1]) ),
    }


def runtrain(args):
    if args.pretrained_ep>=1:
        print("not implemented")#args, model = get_model_ckpt(args)
    else:
        its = load_data(args)
        model_ = get_model(args)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = get_optims(args, model_)

    if args.lrschedule=='rop':
        scheduler = get_scheduler(args, optimizer)

    trainer = get_trainer(args, model_, loss_fn, optimizer)
    evaluator = get_eval(args, model_, loss_fn)
    logger = get_logger(args)

    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        pprint(optimizer)
        pprint(loss_fn)
        pprint(args)
        print(f"SOS, EOS, PAD tokens are as follows: {SOS_TOKEN, EOS_TOKEN, PAD_TOKEN}")
        print(f"MAX and MIN tokens = 658, 0")

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        log_results(logger, 'train/iter', engine.state, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_epoch(engine):
        log_results(logger, 'train/epoch', engine.state, engine.state.epoch)
        evalstate = runeval(evaluator, its['val'])
        log_results(logger, 'val/epoch', evaluator.state, engine.state.epoch)

        if engine.state.epoch % args.save_every==0 and engine.epoch>0:
            save_ckpt(args, model_, None, engine.state.epoch, evalstate.metrics['loss'])
