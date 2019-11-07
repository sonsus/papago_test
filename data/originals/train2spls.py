from pathlib import Path
from random import randrange
import jsonlines as jsl

src_o, trg_o = "./train_source.txt", "./train_target.txt"
wr_t, wr_v = "train.jsonl", "val.jsonl"

def numericalize(numberstream):
    return [int(n) for n in numberstream.rstrip().split()]

## train ==> train / val
with Path(src_o).open() as read_s, Path(trg_o).open() as read_o, \
    jsl.open(wr_t, mode='w') as write_train, jsl.open(wr_v, mode='w') as write_val:

    for i, (ls, lt) in enumerate( zip ( read_s.readlines(), read_o.readlines() ) ) :
        obj = {'src': numericalize(ls), 'trg': numericalize(lt) }
        if i%9 == randrange(0,9):
            write_val.write(obj)
        else:
            write_train.write(obj)


##merge test
src_test, trg_test = "test_source.txt", "test_target.txt"

with Path(src_test).open() as ts, Path(trg_test).open() as tt, jsl.open('test.jsonl', mode='w') as write_test:
    for ls, lt in zip(ts.readlines(), tt.readlines() ):
        obj = {'src': numericalize(ls), 'trg': numericalize(lt) }
        write_test.write(obj)
