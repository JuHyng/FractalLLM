wandb login 20f894088a42a42e5eef02b48b1e6cce6805fdfe
wandb online

for dataset in human_eval
do
python main.py \
    --model_name codellama/CodeLlama-13b-Instruct-hf \
    --draft_token "[DRAFT{i}]" \
    --decode_method baseline \
    --decomp_method quant_8bit \
    --draft_len 4 \
    --draft_layer_indexes 6 10 11 15\
    --split test \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 500 \
    --num_beams 1 \
    --n_fewshot 0 \
    --use_cache False \
    --print_draft False \
    --device_map "auto" 
done

for dataset in human_eval
do
python main.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --draft_token "[DRAFT{i}]" \
    --decode_method fractal \
    --decomp_method quant_8bit \
    --draft_len 8 \
    --draft_layer_indexes 3 7 11 15 19 23 27 31\
    --split test \
    --output_dir ./ \
    --dataset $dataset \
    --max_samples 500 \
    --num_beams 1 \
    --n_fewshot 0 \
    --use_cache False \
    --print_draft False \
    --device_map "auto" 
done

