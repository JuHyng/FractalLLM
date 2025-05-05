wandb online

for dataset in gsm8k xsum
do
python main.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --draft_token "[DRAFT{i}]" \
    --decode_method fractal \
    --decomp_method quant_8bit \
    --draft_len 8 \
    --draft_layer_indexes 5 10 15 20 25 30\
    --split train \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 2 \
    --use_cache False
done