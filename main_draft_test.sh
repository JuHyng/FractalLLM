wandb offline

for dataset in cnn_dm
do
python main.py \
    --model_name meta-llama/Llama-3.2-1B \
    --decode_method draft \
    --decomp_method quant_8bit \
    --draft_len 8 \
    --split train \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 0 \
    --device_map "cuda:1" \
    --use_cache False
done