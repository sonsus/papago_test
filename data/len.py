import jsonlines as jsl
import math
from pdb import set_trace

src_lens = []
trg_lens = []

def std(lens):
    mean = sum(lens)/len(lens)
    return math.sqrt(sum([ (s-mean)**2 for s in lens])/len(lens))


for n in ['train', 'val', 'test']:
    with jsl.open(n+".jsonl") as reader:
        for obj in reader:
            src_lens.append(len(obj['src']))
            trg_lens.append(len(obj['trg']))
    print(f"{n} length prepared")
    print(f"max: {max(src_lens), max(trg_lens)}")
    print(f"min: {min(src_lens), min(trg_lens)}")
    print(f"mean: {sum(src_lens)/len(src_lens), sum(trg_lens)/len(trg_lens)}")
    med_src = sorted(src_lens)[len(src_lens)//2]
    med_trg = sorted(trg_lens)[len(trg_lens)//2]
    print(f"med: {med_src, med_trg}")
    std_src = sum([ (s-sum(src_lens)/len(src_lens))**2 for s in src_lens])
    print(f"std: {std(src_lens), std(trg_lens)}")

    src_lens = []
    trg_lens = []
    set_trace()




'''
train length prepared
max: (81, 54)
min: (2, 1)
mean: (19.00154487872702, 10.065039394407538)
med: (16, 9)
std: (12.828763163523032, 7.251326939142525)



val length prepared
max: (68, 44)
min: (2, 1)
mean: (18.852604828462518, 9.944091486658195)
med: (16, 8)
std: (13.112013859286675, 7.3970305544452275)



test length prepared
max: (84, 54)
min: (1, 1)
mean: (19.049, 10.0955)
med: (16, 9)
std: (12.852805102389107, 7.2080080292685595)
'''
