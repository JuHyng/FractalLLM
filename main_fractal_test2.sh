wandb online


for dataset in gsm8k
do
python main2.py \
    --model_name meta-llama/Llama-3.2-1B \
    --draft_token "[DRAFT{i}]" \
    --decode_method fractal \
    --decomp_method quant_8bit \
    --draft_len 4 \
    --draft_layer_indexes 12 13 14 15\
    --split train \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 0 \
    --use_cache False \
    --print_draft True
done