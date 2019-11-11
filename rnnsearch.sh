CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.5 --min_lr 1e-7 > log/rnnsearch_0.5.out &
processid0=$!
sleep 1m
CUDA_VISIBLE_DEVICES=1 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.3 --min_lr 1e-7 > log/rnnsearch_0.3.out &
processid1=$!
wait $processid1

CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.1 --min_lr 1e-7 > log/rnnsearch_0.1.out &
processid2=$!
sleep 1m
CUDA_VISIBLE_DEVICES=1 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.07 --min_lr 1e-7 > log/rnnsearch_0.07.out &
processid3=$!
wait $processid3

CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.05 --min_lr 1e-7 > log/rnnsearch_0.05.out &
processid4=$!
sleep 1m
CUDA_VISIBLE_DEVICES=1 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.03 --min_lr 1e-7 > log/rnnsearch_0.03.out &
processid5=$!
wait $processid5

CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.01 --min_lr 1e-7 > log/rnnsearch_0.01.out &
processid6=$!
sleep 1m
CUDA_VISIBLE_DEVICES=1 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.007 --min_lr 1e-7 > log/rnnsearch_0.007.out &
processid7=$!
wait $processid7
