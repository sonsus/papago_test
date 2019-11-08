for I in 1 2 4 6 8 10
do
    CUDA_VISIBLE_DEVICES=1 nohup python main.py train --model seq2seq  --max_epoch 100 --lrschedule noam --factor_tr $I --warmup 6900 > log/seq2seq_noam_$I.out &
    processid1=$!
    wait $processid0
done
