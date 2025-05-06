wandb offline

for dataset in cnn_dm
do
python main.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --draft_token "[DRAFT{i}]" \
    --decode_method fractal \
    --decomp_method quant_8bit \
    --draft_len 8 \
    --draft_layer_indexes 24 25 26 27 28 29 30 31\
    --split train \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 0 \
    --use_cache False \
    --print_draft True
done