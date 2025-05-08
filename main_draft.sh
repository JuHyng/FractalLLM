wandb online

for dataset in human_eval
do
python main.py \
    --model_name meta-llama/Llama-3.2-1B \
    --decode_method draft \
    --decomp_method quant_8bit \
    --draft_len 8 \
    --split test \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 0 \
    --device_map "cuda:0" \
    --use_cache False
done