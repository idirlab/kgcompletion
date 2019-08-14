CUDA_VISIBLE_DEVICES=0 python main.py model ConvE dataset FB15k \
                                      input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 \
                                      lr 0.003 process True
