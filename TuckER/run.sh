#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15k --num_iterations 500 --batch_size 128 --lr 0.003 --dr 0.99 \
#                                      --edim 200 --rdim 200 --input_dropout 0.2 --hidden_dropout1 0.2 \
#                                      --hidden_dropout2 0.3 --label_smoothing 0.0 > log/TuckER-FB15k.log

CUDA_VISIBLE_DEVICES=1 python main.py --dataset FB15k-237 --num_iterations 500 --batch_size 128 --lr 0.0005 --dr 1.0 \
                                      --edim 200 --rdim 200 --input_dropout 0.3 --hidden_dropout1 0.4 \
                                      --hidden_dropout2 0.5 --label_smoothing 0.1 > log/TuckER-FB15k-237.log

#CUDA_VISIBLE_DEVICES=0 python main.py --dataset WN18 --num_iterations 500 --batch_size 128 --lr 0.005 --dr 0.995 \
#                                      --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.1 \
#                                      --hidden_dropout2 0.2 --label_smoothing 0.1 > log/TuckER-WN18.log

#CUDA_VISIBLE_DEVICES=0 python main.py --dataset WN18RR --num_iterations 500 --batch_size 128 --lr 0.01 --dr 1.0 \
#                                      --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.2 \
#                                      --hidden_dropout2 0.3 --label_smoothing 0.1 > log/TuckER-WN18RR.log

CUDA_VISIBLE_DEVICES=0 python main.py --dataset YAGO3-10 --num_iterations 500 --batch_size 128 --lr 0.005 --dr 1.0 \
                                      --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.2 \
                                      --hidden_dropout2 0.3 --label_smoothing 0.1 > log/TuckER-YAGO3-10.log

CUDA_VISIBLE_DEVICES=1 python main.py --dataset YAGO3-10 --num_iterations 500 --batch_size 128 --lr 0.005 --dr 1.0 --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.2 --hidden_dropout2 0.3 --label_smoothing 0.1 > log/TuckER-YAGO3-10.log
