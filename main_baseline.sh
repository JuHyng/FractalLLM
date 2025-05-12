wandb online

for dataset in gsm8k xsum
do
python main.py \
    --model_name meta-llama/Llama-2-13b-hf \
    --decode_method baseline \
    --split test \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 0 \
    --device_map "auto" \
    --use_cache False
done

# meta-llama/Llama-3.2-1B