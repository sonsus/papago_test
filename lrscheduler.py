import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


from pdb import set_trace

def get_scheduler(args, optimizer=None):
    if args.lrschedule=="rop":
        return ReduceLROnPlateau(optimizer,
                                mode= args.mode,
                                factor= args.factor,
                                patience= args.patience,
                                threshold=args.threshold,
                                threshold_mode= args.threshold_mode,
                                min_lr=args.min_lr,
                                eps=args.eps,
                            )
    else:
        print("noam opt is implemented as an optimizer class (optims.py)")
        return None
