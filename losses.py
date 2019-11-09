import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


from config import * #SPECIAL TOKENS

def get_loss(ignore_index=PAD_TOKEN, smooth=0):
    return CELoss(ignore_index=ignore_index, smooth=smooth)



class CELoss(CrossEntropyLoss):
    def __init__(self, ignore_index=0, smooth=0):
        super().__init__(ignore_index=ignore_index)
        self.smooth = smooth

    def forward(self, pred, trg):
        if pred.nelement() == 0 or trg.nelement()==0:
            return None
        pred = pred.contiguous().view(-1, pred.shape[-1])
        trg = trg.contiguous().view(-1)

        if self.smooth>0:
            n_class = pred.shape[1]
            onehot = torch.zeros_like(pred).scatter(1,trg.view(-1,1), 1)
            onehot = onehot *( 1- self.smooth) + (1-onehot) *self.smooth / (n_class-1)  # -1 for original target
            logprob = F.log_softmax(pred, dim=1)

            donotpad = (trg!=PAD_TOKEN)
            loss = -(onehot * logprob).sum(dim=1)
            loss = loss.masked_select(donotpad).mean()
        else:
            loss = super().forward(pred, trg)

        return loss
