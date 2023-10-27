# Sample script to finetune RAG using Ray for distributed retrieval.

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

#export MODEL_NAME_OR_PATH='facebook/rag-token-base'
export MODEL_NAME_OR_PATH='facebook/rag-token-base'
export OUTPUT_DIR='ragae_rac/'
export DATA_DIR='msmarco/'
# Start a single-node Ray cluster.
ray start --head

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag_ray.sh --help to see all the possible options

python finetune_ragk.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type ragk_token \
    --fp16 \
    --gpus 1 \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1 \
    --train_batch_size 8 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 50 \
    --val_max_target_length 50 \
    --test_max_target_length 100 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-06 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 1e-05 \
    --num_train_epochs 100 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --distributed_retriever ray \
    --num_retrieval_workers 4 \
    --early_stopping_patience 3 \
    --ragae_type rac \
    --index_name custom \
    --passages_path msmarco_ks/my_knowledge_dataset \
    --index_path msmarco_ks/my_knowledge_dataset_hnsw_index.faiss \
    --n_docs 10 \

# Stop the Ray cluster.
ray stop
