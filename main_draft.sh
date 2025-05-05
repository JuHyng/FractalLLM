wandb online

for dataset in xsum gsm8k 
do
python main.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --decode_method draft \
    --decomp_method quant_8bit \
    --draft_len 8 \
    --split train \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 2 \
    --use_cache False
done