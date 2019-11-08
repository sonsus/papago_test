import torchtext
#from torchtext.datasets import TranslationDataset
from torchtext.data import Dataset, Iterator, Field, TabularDataset
from pathlib import Path
from config import *

from pdb import set_trace


'''def tokenizer(x):
    return [s for s in x.rstrip().split()]
'''
def load_data(args):
    print(f"Start reading and get iterators ")
    src = Field(sequential=True,
                    use_vocab=False,
                    #tokenize=tokenizer,
                    #preprocessing= lambda x: [int(i) for i in x ],
                    init_token=SOS_TOKEN,
                    pad_token=PAD_TOKEN,
                    )
    trg = Field(sequential=True,
                    use_vocab=False,
                    #tokenize=tokenizer,
                    #preprocessing= lambda x: [int(i) for i in x ],
                    eos_token=EOS_TOKEN,
                    pad_token=PAD_TOKEN,
                    )
    splts = ['train', 'val', 'test']

    train, val, test = TabularDataset.splits(path = Path(args.dataroot),
                                                format = 'json',
                                                train='train.jsonl', validation='val.jsonl', test='test.jsonl',
                                                fields ={
                                                    'src': ('src', src),
                                                    'trg': ('trg', trg)
                                                } )

    iterators_splts = Iterator.splits(
                                        (train,val,test),
                                        batch_size=args.batchsize,
                                        device= args.device,
                                        sort_within_batch=True,
                                        sort_key=lambda x: len(x.trg),
                                        shuffle=args._training,
                                        )

    return dict(zip(splts, iterators_splts))
