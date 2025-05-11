wandb login 20f894088a42a42e5eef02b48b1e6cce6805fdfe
wandb online

for dataset in gsm8k xsum cnn_dm human_eval
do
python main.py \
    --model_name meta-llama/Llama-3.2-1B \
    --draft_token "[DRAFT{i}]" \
    --decode_method draft \
    --decomp_method quant_8bit \
    --draft_len 1 \
    --draft_layer_indexes 6 10 11 15\
    --split test \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 1000 \
    --num_beams 1 \
    --n_fewshot 0 \
    --use_cache False \
    --print_draft True \
    --device_map "cuda:1" 
done