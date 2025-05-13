import logging
import time
from collections import deque

import torch
import random
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from src.args import get_args
from src.dataset import load_dataset
from src.model import load_model_with_fractal, set_verify_mode, load_quantized_model
from src.generate import ParallelSPGenerator
from src.utils import FlopsCounter


def prepare_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    draft_tokens = None
    if args.decode_method == 'fractal':
        if args.draft_token == "[DRAFT]":
            draft_tokens = ["[DRAFT]"] * args.draft_len
            tokenizer.add_special_tokens({"additional_special_tokens": draft_tokens})
        elif args.draft_token == "[DRAFT{i}]":
            draft_tokens = [f"[DRAFT{i}]" for i in range(1, args.draft_len + 1)]
            tokenizer.add_special_tokens({"additional_special_tokens": draft_tokens})
        elif args.draft_token != "unk":
            draft_tokens = [args.draft_token] * args.draft_len

    return tokenizer, draft_tokens


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)

    ts = time.strftime("%y_%m-%d_%H:%M", time.localtime())
    if args.decode_method == "fractal":
        exp_name = (
            f"{ts}-{args.model_name}-{args.decode_method}-{args.dataset}-max_samples:{args.max_samples}-"
            f"{args.decomp_method}-draft_len:{args.draft_len}-draft_token:{args.draft_token}"
        )
    elif args.decode_method == "draft":
        exp_name = (
            f"{ts}-{args.model_name}-{args.decode_method}-{args.dataset}-max_samples:{args.max_samples}-"
            f"draft_len:{args.draft_len}"
        )
    else:
        exp_name = f"{ts}-{args.model_name}-{args.decode_method}-{args.dataset}-max_samples:{args.max_samples}"

    wandb.init(project="FractalLLM", entity='kimjuhyng', name=exp_name, config=args.__dict__)
    
    if args.sweep:
        wandb.define_metric("elapsed", summary="mean")

        cfg_dict = dict(vars(args))
        cfg_dict.update(wandb.config)      # sweep 값으로 덮어쓰기

        # (A) 0/1 플래그 → 선택된 레이어 인덱스 목록 만들기
        layer_flags = [cfg_dict.get(f"layer_{i}", 0)
                       for i in range(args.draft_len * 4)]  # 예: draft_len=4면 16플래그
        layer_idxs = [i for i, flag in enumerate(layer_flags) if int(flag) == 1]

        # (B) 선택 수가 draft_len 보다 많으면 랜덤으로 잘라내기 ★수정
        max_select = args.draft_len
        if len(layer_idxs) > max_select:
            layer_idxs = random.sample(layer_idxs, max_select)

        # (C) args.draft_layer_indexes 갱신
        cfg_dict["draft_layer_indexes"] = layer_idxs
        setattr(args, "draft_layer_indexes", layer_idxs)

        print(f"[Sweep] draft_layer_indexes → {args.draft_layer_indexes}")
        wandb.config.update(cfg_dict, allow_val_change=True)
        setattr(args, "draft_layer_indexes", layer_idxs)

    # 데이터 및 토크나이저 준비
    data_iter, max_length = load_dataset(args)
    data_list = list(data_iter)
    tokenizer, draft_tokens = prepare_tokenizer(args)

    # 모델 로드
    if args.decode_method == "fractal":
        model = load_model_with_fractal(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            tokenizer=tokenizer,
            device_map=args.device_map,
            experiment_config=args,
        )
        draft_mode_func = set_verify_mode
    elif args.decode_method == 'baseline':
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
        )
        draft_mode_func = None
    elif args.decode_method == 'draft':
        target_model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
        )
        draft_model = load_quantized_model(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            experiment_config=args,
        )
        draft_mode_func = None


    # 성능 기록용 dicts
    perf_base = {"times": [], "flops": [], "tokens": []}
    perf_spec = {"times": [], "flops": [], "tokens": []}

    if args.decode_method == "baseline":
        model.eval()
        flops_counter = FlopsCounter(model)
        
        
        for idx, (question, _) in enumerate(tqdm(data_list, desc="baseline")):
            flops_counter.reset()

            enc = tokenizer(question, return_tensors="pt").to(model.device)
            generated = enc["input_ids"].clone()
            prompt_len = generated.size(1)
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("토크나이저에 EOS 토큰이 지정되어 있지 않습니다.")

            t0 = time.time()
            with torch.no_grad():
                for _ in range(max_length):
                    logits = model(generated, use_cache=args.use_cache).logits[:, -1, :]
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                    ### EOS
                    if next_id.item() == tokenizer.eos_token_id:
                        break
                    generated = torch.cat([generated, next_id], dim=-1)

            elapsed = time.time() - t0

            flops = flops_counter.get_total_flops()
            new_tokens = generated.size(1) - prompt_len

            perf_base["times"].append(elapsed)
            perf_base["flops"].append(flops)
            perf_base["tokens"].append(new_tokens)

            wandb.log({
                "mode":      "baseline",
                "idx":       idx,
                "elapsed":   elapsed,
                "flops":     flops,
                "new_tokens": new_tokens,
            })

        wandb.log({
            "baseline/total_time": sum(perf_base["times"]),
            "baseline/avg_time":   sum(perf_base["times"]) / len(perf_base["times"]),
            "baseline/total_flops": perf_base["flops"],
            "baseline/total_tokens": perf_base["tokens"],
        })
        
    elif args.decode_method == "draft":
        target_model.eval()
        draft_model.eval()
        
        flops_counter_t = FlopsCounter(target_model)
        flops_counter_d = FlopsCounter(draft_model)
        
        for idx, (question, _) in enumerate(tqdm(data_list, desc="draft")):
            performance_dict = {
                "model_forward_count": {"draft": 0, "verify": 0},
                "new_tokens": 0,
                "draft_time": 0.0,
                "verify_time": 0.0,
                "accept_ratio": 0.0,
            }
            
            flops_counter_t.reset()
            flops_counter_d.reset()
            accept_count = 0
            total_checks = 0
            
            enc = tokenizer(question, return_tensors="pt").to(draft_model.device)
            generated = enc["input_ids"].clone()
            prompt_len = generated.size(1)
            t0 = time.time()
            
            done=False
            with torch.no_grad():
                total_generated_tokens = 0
                while not done:
                    draft_seq = generated.clone()
                    
                    # draft
                    t_draft_start = time.time()
                    for i in range(args.draft_len):
                        if total_generated_tokens + i >= max_length:
                            break
                        draft_logits = draft_model(draft_seq, use_cache=args.use_cache).logits[:, -1, :]
                        next_id = torch.argmax(draft_logits, dim=-1, keepdim=True)
                        if next_id.item() == tokenizer.eos_token_id:
                            break
                        draft_seq = torch.cat([draft_seq, next_id], dim=-1)
                        performance_dict["model_forward_count"]["draft"] += 1
                        
                    t_draft = time.time() - t_draft_start
                    performance_dict["draft_time"] += t_draft
                        
                    old_len = generated.size(1)
                    new_len = draft_seq.size(1)
                    num_drafted = new_len - old_len
                    total_checks += num_drafted
                    
                    ### DEBUG
                    print("total_generated_tokens:", total_generated_tokens)

                    if num_drafted > 0:
                        t_verify_start = time.time()
                        # Target model forward pass (assuming use_cache works correctly)
                        verify_logits = target_model(draft_seq, use_cache=args.use_cache).logits[:, old_len - 1:, :]
                        verify_next_ids = torch.argmax(verify_logits, dim=-1) # Shape: [batch_size, num_drafted + 1]
                        performance_dict["model_forward_count"]["verify"] += 1
                        t_verify = time.time() - t_verify_start
                        performance_dict["verify_time"] += t_verify

                        accepted_len = 0
                        for i in range(num_drafted):
                            draft_token_at_i = draft_seq[:, old_len + i]
                            target_pred_at_i = verify_next_ids[:, i]
                            
                            if torch.equal(draft_token_at_i, target_pred_at_i):
                                accept_count += 1
                                accepted_len += 1
                                ### DEBUG (Optional)
                                print(f"[MATCH]: draft: {draft_token_at_i.item()}, verify: {target_pred_at_i.item()}")
                            else:
                                ### DEBUG (Optional)
                                print(f"[MISMATCH]: draft: {draft_token_at_i.item()}, verify: {target_pred_at_i.item()}")
                                break # Stop at the first mismatch
                            
                            if target_pred_at_i == tokenizer.eos_token_id:
                                done = True
                                break

                        # Append accepted tokens (if any)
                        if accepted_len > 0:
                            accepted_tokens = draft_seq[:, old_len : old_len + accepted_len]
                            generated = torch.cat([generated, accepted_tokens], dim=-1)
                            total_generated_tokens += accepted_len

                        # Append the corrected token if there was a mismatch AND we haven't reached max_length
                        if accepted_len < num_drafted and total_generated_tokens < max_length:
                            # Use the target model's prediction at the mismatch point (index accepted_len)
                            corrected_token = verify_next_ids[:, accepted_len : accepted_len + 1]
                            generated = torch.cat([generated, corrected_token], dim=-1)
                            total_generated_tokens += 1
                        
                        # DEBUG
                        # print("total_generated_tokens:", total_generated_tokens)
                        # print("new_tokens:", generated.size(1) - prompt_len)

                    # Check if generation finished after this draft/verify cycle
                    if total_generated_tokens >= max_length:
                        done = True

                    ### REMOVE DEBUG input()
                    text_out = tokenizer.decode(generated[0], skip_special_tokens=False)
                    # print(f"[{idx}] Intermediate Generated >>> {text_out}")
                    
            accept_ratio = accept_count / total_checks if total_checks > 0 else 0.0
            elapsed = time.time() - t0
            flops_t = flops_counter_t.get_total_flops()
            flops_d = flops_counter_d.get_total_flops()
            new_tokens = generated.size(1) - prompt_len

            text_out = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"[{idx}] Generated >>> {text_out}")

            perf_spec["times"].append(elapsed)
            perf_spec["tokens"].append(new_tokens)

            performance_dict["new_tokens"] = new_tokens
            performance_dict["accept_ratio"] = accept_ratio

            log_data = {
                "mode":       "draft",
                "elapsed":    elapsed,
                "flops_t":   flops_t,
                "flops_d":   flops_d,
                "flops": flops_t + flops_d,
                "new_tokens": new_tokens,
                **performance_dict,
            }
            wandb.log(log_data)
            
            
    elif args.decode_method == "fractal": 
        model.eval()
        flops_counter = FlopsCounter(model)
        
        for data in tqdm(data_list, desc="fractal"):
            
            performance_dict = {
                "model_forward_count": {"draft": 0, "verify": 0},
                "new_tokens": 0,
                "draft_time": 0.0,
                "verify_time": 0.0,
                # ==================== NEW ====================
                "total_accept_count": 0,    # 누적 맞은 토큰 수
                "total_checked_count": 0,   # 누적 검증한 토큰 수
                "accept_ratio": 0.0,        # 실시간 비율
                # ============================================
            }
            
            gen = ParallelSPGenerator(
                model=model,
                tokenizer=tokenizer,
                draft_mode_func=draft_mode_func,
                data_queue=deque([data]),
                args=args,
                draft_tokens=draft_tokens,
                performance_dict=performance_dict,
                max_length=max_length,
            )

            flops_counter.reset()
            t0 = time.time()
            with torch.no_grad():
                performance_dict = gen.forward()
            elapsed = time.time() - t0
            flops = flops_counter.get_total_flops()
            new_tokens = performance_dict["new_tokens"]

            perf_spec["times"].append(elapsed)
            perf_spec["flops"].append(flops)
            perf_spec["tokens"].append(new_tokens)

            log_data = {
                "mode":       "speculative",
                "elapsed":    elapsed,
                "flops":      flops,
                "new_tokens": new_tokens,
                **performance_dict,
            }
            wandb.log(log_data)

        wandb.log({
            "fractal/total_time":  sum(perf_spec["times"]),
            "fractal/avg_time":    sum(perf_spec["times"]) / len(perf_spec["times"]),
            "fractal/total_flops": perf_spec["flops"],
            "fractal/total_tokens": perf_spec["tokens"],
            **{f"model_fw_count/{k}": v for k, v in performance_dict["model_forward_count"].items()},
            "draft_time_total":   performance_dict["draft_time"],
            "verify_time_total":  performance_dict["verify_time"],
        })

    wandb.finish()


if __name__ == "__main__":
    main()
