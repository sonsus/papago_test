from torch.nn import CrossEntropyLoss


def get_loss(**kwargs):
    return CELoss().resolve_args(**kwargs)


class CELoss(CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, pred, trg):
        if pred.nelement() == 0 or trg.nelement()==0:
            return None
        pred = pred.contiguous().view(-1, pred.shape[-1])
        trg = trg.contiguous().view(-1)
        loss = super().forward(pred, trg)
        return loss

    @classmethod
    def resolve_args(cls, ignore_index=0):
        options = {}
        options['ignore_index'] = ignore_index

        return cls(**options)
