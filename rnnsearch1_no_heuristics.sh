CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.5 --min_lr 1e-7 > log/rnnsearch_0.5_noh.out &
processid0=$!
sleep 1m
CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model rnnsearch  --label_smoothing 0.2 --max_epochs 100 --learning_rate 0.3 --min_lr 1e-7 > log/rnnsearch_0.3_noh.out &
processid1=$!
wait $processid1
