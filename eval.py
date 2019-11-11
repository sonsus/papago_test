import torch
from ignite.engine.engine import Engine, State, Events
from ignite.metrics import Loss
from pprint import pprint

from ckpt import *
from metrics import Ngram, Ldiff_Square
from utils import *
from losses import get_loss

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
    if args.load_path is not 'none':
        saved_dict = get_model_ckpt(args)
        its= saved_dict['its']
        model_ = saved_dict['model']

        loss_fn = get_loss(ignore_index=PAD_TOKEN, smooth=args.label_smoothing)
        scheduler = None
        evaluator = get_eval(args, model_, loss_fn)

        @evaluator.on(Events.STARTED)
        def on_eval_start(engine):
            pprint(args)
            pprint(args.model)
            pprint(args.load_path)
        @evaluator.on(Events.EPOCH_COMPLETED)
        def after_an_epoch(engine):
            def results(spl, state):
                for key, val in state.metrics.items():
                    if isinstance(val, dict):
                        for key2, v in val.items():
                            print(f"{spl}/{key}/{key2}: {v}" )
                    else:
                        print(f"{name}/{key}" , val)
            #printout all metrics
            results('test', engine.state)
            print(f"trg_eval: {engine.state.output['trg_idx'][0]}")
            print(f"greedy: {engine.state.output['infer']['greedy']['sentidxs'][0]}")
            print(f"beam: {engine.state.output['infer']['beam']['sentidxs'][0]}")

        evaluator.run(its['val'] if args.debug else its['test'],
                    max_epochs=1)


    else:
        print(f"need to specify args.load_path (e.g. .pth file or pth containing dir )")
        print(f"python main.py --load_path trained_models/rnnsearch/rnnsearch_adam_ep100_labelsmooth0.2_d1111_t2037/")
        print(f"if specify dir, it will load the lowest loss model.")
