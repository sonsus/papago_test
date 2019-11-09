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
from optims import get_optims
from losses import get_loss
from ckpt import *

from pprint import pprint

from pdb import set_trace

def get_trainer(args, model_, loss_fn, optimizer, scheduler=None):
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
        loss = loss_fn(y_pred, batch.trg )
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
    for n, metric in metrics.items():
        metric.attach(trainer, n)

    return trainer


def runtrain(args):
    if args.pretrained_ep>=1:
        saved_dict = get_model_ckpt(args)
        its= saved_dict['its']
        model_ = saved_dict['model']
        optimizer = saved_dict['optimizer']
    else:
        its = load_data(args)
        model_ = get_model(args)
        optimizer = get_optims(args, model_)

    loss_fn = get_loss(ignore_index=PAD_TOKEN, smooth=args.label_smoothing)
    scheduler = None

    if args.lrschedule=='rop':
        scheduler = get_scheduler(args, optimizer)

    trainer = get_trainer(args, model_, loss_fn, optimizer, scheduler=scheduler)
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
        log_lr(logger, 'ep-lr', optimizer, engine.state.epoch)
        log_results(logger, 'train/epoch', engine.state, engine.state.epoch)
        evalstate = runeval(evaluator, its['val'])
        log_results(logger, 'val/epoch', evaluator.state, engine.state.epoch)

        if scheduler is not None:
            scheduler.step(evalstate.metrics['loss'])
        print(f"epoch: {engine.state.epoch}  completed")
        print(f"trg: {engine.state.output[1]}")
        print(f"teacher: {engine.state.output[0].argmax(dim=1)}")

        print(f"trg_eval: {evaluator.state.output['trg_idx']}")
        print(f"greedy: {evaluator.state.output['infer']['greedy']['sentidxs'][0]}")
        print(f"beam: {evaluator.state.output['infer']['beam']['sentidxs'][0]}")

        if engine.state.epoch % args.save_every==0 and engine.state.epoch>0:
            save_ckpt(args, model_, None, engine.state.epoch, evalstate.metrics['loss'], optimizer=optimizer)

    trainer.run(its['val'] if args.debug else its['train'],
                max_epochs=args.max_epochs)
