wandb offline

for dataset in cnn_dm
do
python main.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --decode_method baseline \
    --split train \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 0 \
    --use_cache False
done