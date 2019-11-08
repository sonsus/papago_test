from torch.optim.optimizer  import Optimizer
from torch.optim import Adam

def get_optims(args, model):
    if args.lrschedule == 'noam':
        optim = NoamOpt.resolve_args(args, model.parameters())
    else:
        optim = Adam(model.parameters(), lr=args.learning_rate )#AdamOpt.resolve_args(args, model.parameters())
    return optim


class NoamOpt(Adam):
    def __init__(self, params, model_size, factor, warmup,
                lr, weight_decay, betas, eps ):

        defaults = dict(lr=lr, betas= betas, eps=eps, weight_decay=weight_decay, amsgrad=False)
        super(Adam, self).__init__(params, defaults)
        # super().__init__(params, defaults)
        ## above alternate causes problem: lr is passed as a dict (==defaults)
        ## shouldn't this be the same?

        self._step = 0
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        #for p in self.optimizer.param_groups:
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        super().step()
        #self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    @classmethod
    def resolve_args(cls, args, params):
        # NoamOpt is separated with .optimizer.Adam class @ ./optimizer/adam.py
        options = {}

        # these hyperparameters are from annoatated transformer standard option.
        options['lr'] = args.get("lr_tr", 0)
        options['weight_decay'] = args.get("weight_decay", 0)
        options['betas'] =  args.get('betas', (0.9, 0.98))
        options['eps'] = args.get('eps', 1e-9)

        options['model_size'] = args.get("d_model", 512 )
        options['factor'] = args.get("factor_tr", 1)
        options['warmup'] = args.get("warmup", 4000)

        return cls(params, **options)


'''
class AdamOpt(Adam):
    @classmethod
    def resolve_args(cls, args, params):
        options = {}
        options['lr'] = args.get("learning_rate", 0.01)
        options['weight_decay'] = args.get("weight_decay", 0)
        options['betas'] = args.get("betas", (0.9, 0.999))
        options['eps'] = args.get("eps", 1e-8)

        return cls(params, **options)
'''
