wandb offline

# export CUDA_VISIBLE_DEVICES=0

for dataset in cnn_dm
do
python main.py \
    --model_name meta-llama/Llama-3.2-1B \
    --draft_token "[DRAFT{i}]" \
    --decode_method fractal \
    --decomp_method quant_8bit \
    --draft_len 4 \
    --draft_layer_indexes 4 6 8 10\
    --split train \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 100 \
    --num_beams 1 \
    --n_fewshot 0 \
    --use_cache False \
    --print_draft False \
    --device_map "cuda:2" \
    --sweep False 
done


