
wandb online
#xsum cnn_dm human_eval
for dataset in gsm8k xsum
do
python main.py \
    --model_name meta-llama/Llama-2-13b-hf \
    --draft_token "[DRAFT{i}]" \
    --decode_method fractal \
    --decomp_method quant_8bit \
    --draft_len 10 \
    --draft_layer_indexes 3 7 11 15 19 23 27 30 \
    --split test \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 1000 \
    --num_beams 1 \
    --n_fewshot 0 \
    --use_cache False \
    --print_draft True \
    --device_map "auto" 
done