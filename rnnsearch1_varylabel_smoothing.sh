CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --max_epochs 100 --label_smoothing 0.0 --learning_rate 0.5 --min_lr 1e-7 > log/rnnsearch_0.5_noLS.out &
processid0=$!
sleep 1m
CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --max_epochs 100 --label_smoothing 0.0 --learning_rate 0.3 --min_lr 1e-7 > log/rnnsearch_0.3_noLS.out &
processid1=$!
wait $processid1

CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --max_epochs 100 --label_smoothing 0.5 --learning_rate 0.5 --min_lr 1e-7 > log/rnnsearch_0.5_noLS.out &
processid0=$!
sleep 1m
CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --max_epochs 100 --label_smoothing 0.5 --learning_rate 0.3 --min_lr 1e-7 > log/rnnsearch_0.3_noLS.out &
processid1=$!
wait $processid1
