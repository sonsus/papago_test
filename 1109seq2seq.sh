###threshold 0.01

CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model seq2seq  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.3 --min_lr 1e-7 > log/seq2seq_0.3.out &
processid0=$!
sleep 1m
CUDA_VISIBLE_DEVICES=1 nohup python main.py train --model seq2seq  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.1 --min_lr 1e-7 > log/seq2seq_0.1.out &
processid1=$!
wait $processid1

CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model seq2seq  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.01 --min_lr 1e-7 > log/seq2seq_0.01.out &
processid2=$!
sleep 1m
CUDA_VISIBLE_DEVICES=1 nohup python main.py train --model seq2seq  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.007 --min_lr 1e-7 > log/seq2seq_0.007.out &
processid3=$!
wait $processid3

CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model seq2seq  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.003 --min_lr 1e-7 > log/seq2seq_0.003.out &
processid4=$!
sleep 1m
CUDA_VISIBLE_DEVICES=1 nohup python main.py train --model seq2seq  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.001 --min_lr 1e-7 > log/seq2seq_0.001.out &
processid5=$!
wait $processid5
