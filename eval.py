import torch
from ignite.engine.engine import Engine, State, Events
from ignite.metrics import Loss

from metrics import Ngram, Ldiff_Square
from utils import *

### nlg ngram evaluation, loss engine
def get_eval(args, model, loss_fn):
    def infer(evaluator, batch):
        model.eval()
        args._training =  model.training

        batch = prep_batch(args, batch)
        with torch.no_grad():
            if args.model in ['rnnsearch', 'seq2seq']:
                y_pred = model(batch.src, batch.trg)
                y_pred = y_pred
                results = model.inference(batch.src, beamsize= args.beamsize)

            elif args.model == 'transformers':
                pass

            else:
                """not impl"""
                pass


            return {
                    'y_pred': y_pred,
                    'trg_idx': batch.trg,
                    'infer': results,
                    }


    engine_g = Engine(infer)

    metrics_g = {
        'loss': Loss(loss_fn, output_transform=lambda x: (x['y_pred'], x['trg_idx']) ),
        'ngrams_greed': Ngram(args, make_fake_vocab(), output_transform=lambda x:(x['infer']['greedy']['sentidxs'], x['trg_idx'])  ),
        'ngrams_beam': Ngram(args, make_fake_vocab(), output_transform=lambda x:(x['infer']['beam']['sentidxs'], x['trg_idx'])  ),
        'length_greed': Ldiff_Square(args, output_transform=lambda x:(x['infer']['greedy']['sentidxs'], x['trg_idx'])  ),
        'length_beam': Ldiff_Square(args, output_transform=lambda x:(x['infer']['beam']['sentidxs'], x['trg_idx'])  ),
        }
    for name, metric in metrics_g.items():
        metric.attach(engine_g, name)


    return engine_g


def runeval(evaluator, iterator):
    evaluator.run(iterator)
    return evaluator.state

def only_eval(args):
    if args.pretrained_ep>=1:
        saved_dict = get_model_ckpt(args)
        its= saved_dict['its']
        model_ = saved_dict['model']
    else:
        print(f"need to specify args.pretrained_ep (e.g. for ep 7 model,)")
        print(f"python main.py --pretrained_ep 7 --model rnnsearch")
