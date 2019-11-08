for I in 1 0.5 0.3 0.1 0.01 0.001 0.0001
do
    CUDA_VISIBLE_DEVICES=0 nohup python main.py train --model seq2seq  --max_epoch 100 --learning_rate $I --min_lr 1e-7 > log/seq2seq_$I.out &
    processid0=$!
    wait $processid0
done
