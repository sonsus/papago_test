## from https://github.com/skaro94/vtt_qa_pipeline/
from collections import defaultdict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError

from utils import to_string, to_string2, make_fake_vocab
from config import *

import torch
from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from collections import defaultdict
from pdb import set_trace



class Ngram(Metric):
    def __init__(self, args, vocab, output_transform=lambda x: x):
        super(Ngram, self).__init__(output_transform)

        self.vocab = vocab
        self.args = args

        self.tokenizer = PTBTokenizer()
        scorers = {
            'bleu': (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'meteor': (Meteor(),"METEOR"),
            'rouge': (Rouge(), "ROUGE_L"),
            'cider': (Cider(), "CIDEr")
        }
        self.scorers = [v for k, v in scorers.items() if k in args.metrics]

    @staticmethod
    def default_transform(x):
        return (x[-2], x[-1])

    def reset(self):
        self._data = defaultdict(lambda: 0)
        self._num_ex = 0

    def format_string(self, x):
        if self.args.beamsize==1 or self.args._training:
            x = to_string(self.vocab, x)
            return {str(i): [v] for i, v in enumerate(x)}
        else:
            x = to_string2(self.vocab, x)
            return {str(i): [v[0]] for i, v in enumerate(x)}#only best beam results are recorded here

    def update(self, output):
        y_pred, y = output # y == trg
        num_ex = len(y_pred) # y_pred could be list of tensors thus no y_pred.shape
        res = self.format_string(y_pred)
        gts = self.format_string(y)

        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    self._data[m] += sc * num_ex
            else:
                self._data[method] += score * num_ex
        self._num_ex += num_ex

    def compute(self):
        if self._num_ex == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return {k: v * 100 / self._num_ex for k, v in self._data.items()}
    # *100 for bleu, rouge score percentages


class Ldiff_Square(Metric):
    def __init__(self, args, output_transform=lambda x:x):
         super().__init__(output_transform)
         self.args = args
         self._measure = {
            "lendiff2sum": 0
         }
         self._num_ex =0

    def reset(self):
        self._measure = {
            "lendiff2sum": 0
        }
        self._num_ex = 0

    def reformat(self, x):
        if self.args.beamsize==1 or self.args._training or type(x)!= type([]):
            return x

        else:
            return torch.cat(x, 1)#list of tensors [1, len ]

    def get_len(self, x):
        mask = (x!=PAD_TOKEN) & (x!=SOS_TOKEN) & (x!=EOS_TOKEN)
        mask = mask.long()
        lens = mask.sum(dim=1)
        return lens

    def update(self, output):
        decoded, trg = output

        num_ex = len(trg)
        decoded = self.reformat(decoded)
        d_lens = self.get_len(decoded)
        trg_lens = self.get_len(trg)
        lendiff2 = ((d_lens-trg_lens)**2).sum()
        self._measure['lendiff2sum']+=lendiff2
        self._num_ex+= num_ex


    def compute(self):
        return self._measure["lendiff2sum"]/self._num_ex
