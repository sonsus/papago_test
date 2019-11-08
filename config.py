PAD_TOKEN = 659
SOS_TOKEN = 660
EOS_TOKEN = 661

#max(tokens) == 658
#min(tokens) == 0 --> in the middle


config = {

    'debug': False,
    ##general
    'model': "seq2seq", #"transformer", "transformer"
    'dataroot': 'data',
    'max_epochs': 50,
    'pretrained_ep': -1,
    'beamsize': 4, # beamsize>=1, int (==1: greedy search )

    ##checkpoint loading / saving
    'ckpt_path': 'trained_models/',
    'save_every': 4,  #epochs, int


    ##hyperparams
    'batchsize': 128,
    'dropout': 0.3,
    'word_drop': 0.05,

    'metrics': ['bleu', 'meteor', 'cider', 'rouge'],


    ###lr scheduler (ReduceLROnPlateau)
    'learning_rate': 0.001,
    'optimizer': 'adam', #now just for exp naming
    #'lr_decay': 0,

    'lrschedule': 'rop', #noam
    #reduce on plateau
    'factor': 0.5,
    'patience': 2,
    'mode': 'min',
    'threshold':5e-4,
    'threshold_mode': 'rel',
    'min_lr':1e-7,
    'eps':1e-8,
    #noam
    'warmup': 4000,
    'factor_tr': 2,
    'lr_tr': 0, #this lr is learning rate for NoamOpt.

    ###--modelwise args--###
    #'label_smoothing': 0.1,
    'hidden_size': 256,


    ## transformers
        #transformer - model
    'N': 6, #number of transformer block repetition
    'd_ff': 2048, #dimension of feedforward for tr blocks
    'h': 8, #number of heads for multihead attention
    'dropout_tr': 0.1, # dropout_ratio for transformer
    'd_model': 512,
        #transformer - noam

    #'betas':(0.9, 0.98),
    #'eps':1e-9,

    ## logger
    'log_path': './log',
    'device': 'cuda',
    'log_cmd': False,
    #'shots_only': True,

    ## modes (DO NOT manipulate this manually: cli.py takes care of this)
    '_training': True, #True: model.train() is performed
}
