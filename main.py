from munch import Munch
from config import config
from dataset import *
from utils import *
from pathlib import Path
from pprint import pprint
from fire import Fire

from train import runtrain
from eval import runeval, only_eval


from pdb import set_trace


class Main:
    def __init__(self):
        self.defaults = config

    def _default_args(self, **kwargs):
        args = self.defaults
        args.update(kwargs)
        resolve_paths(args)
        resolve_device(args)
        args = Munch(args)
        return args

    def check_data(self, **kwargs):
        args = self._default_args(**kwargs)
        pprint(args)
        its = load_data(args)
        for name in ["train", "val", "test"]:
            print(name)
            for i,e in enumerate(its[name]):
                batch = prep_batch(args, e)
                print(batch)

        set_trace()

    def train(self, **kwargs):
        args = self._default_args(**kwargs)
        with log_time_n_args(args):
            runtrain(args)

    def eval(self, **kwargs):
        args = self._default_args(**kwargs)
        only_eval(args)


def resolve_paths(args):
    for key in args.keys():
        if key[-4:]=='root' or key[-4:]=='path':
            args[key]=Path(args[key])


def resolve_device(args):
    for key in args.keys():
        if key=='device':
            args[key] == torch.device(args[key])


if __name__ == "__main__":
    Fire(Main)
