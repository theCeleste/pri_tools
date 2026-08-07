#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset sst2 \
--eps 1.0 \
--top_k 20 \
--embedding_type glove_840B-300d \
--save_stop_words True
