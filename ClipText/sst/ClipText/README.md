
# How to run ClipText?

 python main.py \
  --dataset sst2 \
  --eps 1.0 \
  --top_k 20 \
  --embedding_type ct_vectors 

# How to run ClipText+?

 python main.py \
  --dataset sst2 \
  --eps 1.0 \
  --top_k 20 \
  --embedding_type ct_vectors \
  --save_stop_words True
