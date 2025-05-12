import torch
import torch.nn as nn
import time
import wandb
from typing import List, Dict
from .model import FractalLlamaForCausalLM
from transformers.cache_utils import DynamicCache, Cache

from src.utils import FlopsCounter, measure_generation_time, FractalCache
from typing import Optional
from copy import deepcopy

def pad_and_combine_slots(
    slots: List[dict],
    pad_token_id: int,
    leftover_len_list: Optional[List[int]] = None,
):
    slot_map = []
    all_input_ids = []
    all_att_masks = []

    # (1) 배치 내 max_len 파악
    total_lens = []
    for i, slot in enumerate(slots):
        cur_len = slot["current_input_ids"].shape[1]
        total_lens.append(cur_len)
    max_len = max(total_lens) if total_lens else 0

    # (2) slot별 처리
    for i, slot in enumerate(slots):
        s_ids = slot["current_input_ids"]  # shape=(1, seq_len_i)
        leftover_len = leftover_len_list[i]
        total_len_i = s_ids.shape[1]

        # [수정] leftover_len이 seq_len_i보다 크면 clamp
        if leftover_len > total_len_i:
            leftover_len = total_len_i

        # (A) input_ids 패딩
        pad_len = max_len - total_len_i
        if pad_len < 0:
            raise ValueError(
                f"pad_and_combine_slots: slot {i} has length={total_len_i} > max_len={max_len}?"
            )

        if pad_len > 0:
            pad_t = torch.full(
                (1, pad_len),
                fill_value=pad_token_id,
                dtype=s_ids.dtype,
                device=s_ids.device
            )
            new_input_ids = torch.cat([s_ids, pad_t], dim=1)  # shape(1, max_len)
        else:
            new_input_ids = s_ids

        # (B) attention_mask 생성
        # leftover 구간=0, new 구간=1
        att_mask = torch.zeros((1, total_len_i), dtype=torch.long, device=s_ids.device)
        new_tokens_len = total_len_i - leftover_len

        if new_tokens_len < 0:
            raise ValueError(
                f"leftover_len_list[{i}]={leftover_len} > current_input_ids length={total_len_i}"
            )

        if new_tokens_len > 0:
            att_mask[:, leftover_len:] = 1

        # (C) 마스크 패딩
        if pad_len > 0:
            pad_mask = torch.zeros((1, pad_len), dtype=att_mask.dtype, device=s_ids.device)
            att_mask = torch.cat([att_mask, pad_mask], dim=1)  # shape(1, max_len)

        all_input_ids.append(new_input_ids)   # shape(1, max_len)
        all_att_masks.append(att_mask)        # shape(1, max_len)
        slot_map.append(i)

    # (3) 배치 합치기
    if len(all_input_ids) == 0:
        # no slots
        batch_input_ids = torch.zeros((0,0), dtype=torch.long)
        batch_attention_mask = torch.zeros((0,0), dtype=torch.long)
        max_len = 0
    else:
        batch_input_ids = torch.cat(all_input_ids, dim=0)       # (B, max_len)
        batch_attention_mask = torch.cat(all_att_masks, dim=0)  # (B, max_len)

    return batch_input_ids, batch_attention_mask, slot_map, max_len



def draft_forward(
    args,
    slots: List[dict],
    model: FractalLlamaForCausalLM,
    tokenizer,
    draft_mode_func,
    performance_dict: dict, 
    prefix_len_list: List[int]=None,
    past_key_values:Cache=None,
    fractal_key_values_list:List[Cache]=None,
):
    draft_mode_func(model, is_verify=False)
    
    
    if args.use_cache and past_key_values is not None:
        cache_batch_split = past_key_values.batch_split(len(prefix_len_list), 1)
        leftover_len_list = [cbs.get_seq_length() for cbs in cache_batch_split]
    else:
        leftover_len_list = [0]*len(prefix_len_list)

    batch_input_ids, batch_attention_mask, slot_map, max_len = pad_and_combine_slots(
        slots,
        pad_token_id=tokenizer.pad_token_id,
        leftover_len_list=leftover_len_list
    )
    
    target_device = next(model.parameters()).device      # ex) cuda:2
    batch_input_ids      = batch_input_ids.to(target_device, non_blocking=True)
    batch_attention_mask = batch_attention_mask.to(target_device, non_blocking=True)
    
    start_t = time.time()

    if args.use_cache:
        past_seen_tokens = past_key_values.get_seq_length()
        seq_len_new = batch_input_ids.size(1)     
        
        cache_position = torch.arange(
            past_seen_tokens, 
            past_seen_tokens + seq_len_new, 
            device=batch_input_ids.device
        )
    
    eos_token_id = tokenizer.eos_token_id    

    with torch.no_grad():
        outputs = model.forward(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            use_cache=args.use_cache,
            past_key_values=past_key_values if args.use_cache else None,
            fractal_key_values_list=fractal_key_values_list if args.use_cache else None,
            draft_len_current = max(s["final_draft_len"] for s in slots),
            prefix_len_list=prefix_len_list,
            cache_position=cache_position if args.use_cache else None,
        )
    logits = outputs.logits
            
    for batch_idx in range(len(prefix_len_list)):
        prefix_len = prefix_len_list[batch_idx]
        
        slots[batch_idx]["input_ids"] = slots[batch_idx]["input_ids"] [:, :prefix_len]
        
        for i in range(prefix_len, prefix_len + (slots[batch_idx]["final_draft_len"]*2)+1):
            
            token_logits = logits[batch_idx, i-1, :]
            token_prob = token_logits
            token_pred = torch.argmax(token_prob).item()
            
            slots[batch_idx]["input_ids"] = torch.cat([slots[batch_idx]["input_ids"], torch.tensor([[token_pred]], device=slots[batch_idx]["input_ids"].device)], dim=-1) 
            
            if token_pred == eos_token_id:
                slots[batch_idx]["final_draft_len"] = i - prefix_len
                break 
            
    draft_time = time.time() - start_t
    performance_dict["draft_time"] += draft_time        
    
    return slots, outputs

def verify_forward(
    args,
    slots: List[dict],
    model: FractalLlamaForCausalLM,
    tokenizer,
    draft_mode_func,
    prefix_len_list: List[int],
    performance_dict: dict,
    past_key_values: Optional[Cache] = None,
    max_length: Optional[int] = None,
):
    draft_mode_func(model, is_verify=True)
    
    start_t = time.time()
    
    active_idx = [i for i, s in enumerate(slots) if s["continue_draft"]]
    if not active_idx:
        return slots, None, 0
    
    if args.use_cache and past_key_values is not None:
        cache_batch_split = past_key_values.batch_split(len(prefix_len_list), 1)
        leftover_len_list = [cbs.get_seq_length() for cbs in cache_batch_split]
    else:
        leftover_len_list = [0] * len(prefix_len_list)
        
    slots = [slots[i] for i in active_idx]

    if len(slots) == 0:
        return slots, None, 0

    batch_input_ids, batch_attention_mask, slot_map, max_len = pad_and_combine_slots(
        slots,
        pad_token_id=tokenizer.pad_token_id,
        leftover_len_list=leftover_len_list
    )
    target_device = next(model.parameters()).device  # ex) cuda:2
    batch_input_ids = batch_input_ids.to(target_device, non_blocking=True)
    batch_attention_mask = batch_attention_mask.to(target_device, non_blocking=True)
    
    eos_token_id = tokenizer.eos_token_id  # EOS token ID
    
    if args.use_cache:
        past_seen_tokens = past_key_values.get_seq_length()
        seq_len_new = batch_input_ids.size(1)     
        
        cache_position = torch.arange(
            past_seen_tokens, 
            past_seen_tokens + seq_len_new, 
            device=batch_input_ids.device
        )
        
    with torch.no_grad():
        outputs = model.forward(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            use_cache=args.use_cache,
            past_key_values=past_key_values if args.use_cache else None,
            cache_position=cache_position if args.use_cache else None,
        )
        
    logits = outputs.logits
    
    correct_predictions_per_draft = []  # 각 드래프트별 정확한 예측 수
    total_predictions_per_draft = []   # 각 드래프트별 총 예측 수
    
    draft_performance_logs = []  # 각 드래프트의 성능 로그를 저장할 리스트
    
    for i, slot_data in enumerate(slots):
        verified_count   = 0
        fail_draft       = False
        first_fail_label = None
        early_stop = False
        
        total_draft_tokens = slot_data["final_draft_len"] * 2 + 1
        seq_len_current    = slot_data["current_input_ids"].shape[1]

        for step in range(total_draft_tokens):
            pos = seq_len_current - total_draft_tokens + step # 현재 위치가 seq_len +1 해야 이전 기반으로 맞출수있기떄문에
            if pos - 1 < 0 or pos - 1 >= logits.shape[1]:
                break
            token_logits   = logits[i, pos - 1, :]
            verify_pred_id = torch.argmax(token_logits).item()
            curr_id        = slot_data["current_input_ids"][0, pos].item()

            # ───────── MATCH ─────────
            if verify_pred_id == curr_id:
                print(f"Draft {i} - Step {step}: Match! Pred: {verify_pred_id}, Curr: {curr_id}")
                if verify_pred_id == eos_token_id:           # EOS를 맞힌 경우
                    verified_count += 1 #EOS도 맞췄췄으니깐 +1
                    
                    eos_pos = pos
                    slot_data["input_ids"]         = slot_data["input_ids"][:, :eos_pos]
                    slot_data["current_input_ids"] = slot_data["current_input_ids"][:, :eos_pos]
                    slot_data["total_new_tokens"]  = eos_pos - slot_data["prompt_len"]
                    slot_data["continue_draft"] = False

                    early_stop = True
                    # ⑤ **즉시** 외부 루프 종료
                    break
                
                verified_count += 1
                continue                                     # 다음 토큰 검사

            # ───────── MISMATCH ──────
            fail_draft       = True
            first_fail_label = verify_pred_id
            break                      # loop 탈출
        # ------- loop end -------
        if early_stop :
            break
        
        if fail_draft and first_fail_label is not None:
            pre_ids      = slot_data["input_ids"][:, :-(total_draft_tokens - verified_count)]
            if first_fail_label == eos_token_id:
                slot_data["input_ids"] = pre_ids           # EOS 이전까지만 보존
                slot_data["continue_draft"] = False        # 더 이상 draft 안 함
            else:
                # ② 일반 토큰이면 기존 복구 로직 유지
                label_tensor = torch.tensor([[first_fail_label]],
                                            device=pre_ids.device, dtype=pre_ids.dtype)
                slot_data["input_ids"] = torch.cat([pre_ids, label_tensor], dim=-1)
                slot_data["continue_draft"] = True         # 다시 draft            # 복구 후 다시 드래프트
        else:
            # 전부 맞았지만 EOS는 아님 → 계속 드래프트
            slot_data["continue_draft"] = True
        slot_data["total_new_tokens"] = (
            slot_data["input_ids"].size(1) - slot_data["prompt_len"]
        )

        if slot_data["total_new_tokens"] >= max_length:
            # 초과분 잘라내기
            trim     = slot_data["total_new_tokens"] - max_length   # 자를 개수
            new_end  = slot_data["input_ids"].size(1) - trim        # 남길 길이

            slot_data["input_ids"]         = slot_data["input_ids"][:, :new_end]
            slot_data["current_input_ids"] = slot_data["current_input_ids"][:, :new_end]

            slot_data["total_new_tokens"]  = max_length
            slot_data["continue_draft"]    = False
                # ==================== NEW ====================
        performance_dict["total_accept_count"]  += verified_count
        performance_dict["total_checked_count"] += total_draft_tokens
        # ============================================

            
    verify_time = time.time() - start_t
    performance_dict["verify_time"] += verify_time
    performance_dict["new_tokens"] = slots[0]["total_new_tokens"]

    # # 각 드래프트의 성능 로그를 wandb에 로깅
    # for log in draft_performance_logs:
    #     wandb.log({
    #         f"draft_{log['draft_idx']}_accept_ratio": log["accept_ratio"],
    #         f"draft_{log['draft_idx']}_correct_predictions": log["correct_predictions"],
    #         f"draft_{log['draft_idx']}_total_predictions": log["total_predictions"],
    #     })

    return slots, outputs, verified_count



        

class ParallelSPGenerator(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        draft_mode_func,
        args,
        data_queue,
        draft_tokens,
        max_length=None,
        performance_dict=None
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.draft_mode_func = draft_mode_func
        self.args = args
        self.data_queue = data_queue
        self.draft_tokens = draft_tokens
        self.performance_dict = performance_dict
        self.max_length = max_length

        self.slots = []
        for i in range(args.batch_size):
            self.slots.append(self._init_slot(i))
            
            
    def _init_slot(self, slot_idx):
        return {
            "slot_idx": slot_idx,
            "input_ids": None,
            "current_input_ids":None, # for use_cache=True
            "continue_draft": True,
            "active": False,
            "draft_iteration": 0,
            "total_new_tokens": 0
        }
        
    def _fill_slot(self, slot):

        if len(self.data_queue) == 0:
            slot["active"] = False
            return
        question, answer = self.data_queue.popleft()
        inputs = self.tokenizer(question, return_tensors="pt")
        slot["input_ids"] = inputs["input_ids"]
        slot["current_input_ids"] = inputs["input_ids"]
        slot["active"] = True
        slot["continue_draft"] = True
        slot["draft_iteration"] = 0
        slot["total_new_tokens"] = 0
        slot["final_draft_len"] = 0
        slot["prompt_len"] = slot["input_ids"].shape[1]

        
    def forward(self):
        for s in self.slots:
            self._fill_slot(s)

        done = False
        iteration = 0
        
        if self.args.use_cache:
            past_key_values = DynamicCache()
            fractal_key_values_list = [FractalCache() for _ in range(len(self.args.draft_layer_indexes))]
        else:
            past_key_values = None
            fractal_key_values_list = None
            
        draft_ids = self.tokenizer.convert_tokens_to_ids(self.draft_tokens)
        
        
        while not done:
            active_slots = [s for s in self.slots if s["active"]]
            if len(active_slots) == 0:
                done = True
                break              
            
            prefix_len_list = []
            for slot in active_slots:
                prefix_len_list.append(slot["input_ids"].shape[1])
                slot["draft_iteration"] += 1
                slot["total_new_tokens"] = slot["input_ids"].shape[1] - slot["prompt_len"]
                
                
                remaining = self.max_length - slot["total_new_tokens"]
                if remaining <= 0:                     
                    slot["continue_draft"] = False
                    continue
                final_draft_len = min(self.args.draft_len, remaining)
                draft_ids_trunc = draft_ids[:final_draft_len]
                draft_t = torch.tensor([draft_ids_trunc],
                                    device=slot["input_ids"].device)
                slot["input_ids"] = torch.cat([slot["input_ids"], draft_t], dim=-1)
                
                slot["final_draft_len"] = final_draft_len

                
            
            cache_batch_split = None
            if self.args.use_cache:
                if past_key_values.get_seq_length() > 0:
                    cache_batch_split = past_key_values.batch_split(len(active_slots), 1)
                    for i in range(len(cache_batch_split)):
                        if cache_batch_split[i].get_seq_length() > 0:
                            active_slots[i]["current_input_ids"] = active_slots[i]["input_ids"][:, -(self.args.draft_len+1):] # 1(last token of prefix) + draft_tokens (draft_len+1)
                            

            if cache_batch_split is None:
                for i in range(len(active_slots)):
                    active_slots[i]["current_input_ids"] = active_slots[i]["input_ids"]
            

            iteration += 1
            active_slots, _draft_out = draft_forward(
                slots=active_slots,
                model=self.model,
                tokenizer=self.tokenizer,
                draft_mode_func=self.draft_mode_func,
                args=self.args,
                prefix_len_list=prefix_len_list,
                performance_dict=self.performance_dict,
                past_key_values=past_key_values,
                fractal_key_values_list=fractal_key_values_list,
            )
            self.performance_dict["model_forward_count"]["draft"] += 1
            
            
            
            if self.args.use_cache:
                past_key_values = _draft_out.past_key_values
                fractal_key_values_list = _draft_out.fractal_key_values_list
                
                # DEBUG
                if self.args.print_draft:
                    print("[After Draft] input_ids:", self.tokenizer.decode(active_slots[i]["input_ids"][0]))
                    print("[After Draft] input_ids.shape:", active_slots[i]["input_ids"].shape)
                    print("[After Draft] past_key_values.get_seq_length()", past_key_values.get_seq_length())
                    print("[After Draft] fractal_key_values_list[0].get_seq_length()", fractal_key_values_list[0].get_seq_length())
                
                if past_key_values.get_seq_length() > 0:
                    cache_batch_split = past_key_values.batch_split(len(active_slots), 1)
                    for i in range(len(cache_batch_split)):
                        if cache_batch_split[i].get_seq_length() > 0:
                            active_slots[i]["current_input_ids"] = active_slots[i]["input_ids"][:, -(1+self.args.draft_len*2+1):]
                            cache_batch_split[i].crop(-(1+self.args.draft_len+1))
                            print("[After Draft] current_input_ids:", active_slots[i]["current_input_ids"])
                            print("[After Draft] current_input_ids deode:", self.tokenizer.decode(active_slots[i]["current_input_ids"][0]))
                            print("[After Draft] current_input_ids.shape:", active_slots[i]["current_input_ids"].shape)
                        else:
                            active_slots[i]["current_input_ids"] = active_slots[i]["input_ids"]
                    
            else:
                for i in range(len(active_slots)):
                    active_slots[i]["current_input_ids"] = active_slots[i]["input_ids"]
            
            if self.args.print_draft:
                print("[After Draft] decoded text:", self.tokenizer.decode(active_slots[0]["input_ids"][0]))

            active_slots, _verify_out, verified_count = verify_forward(
                slots=active_slots,
                model=self.model,
                tokenizer=self.tokenizer,
                draft_mode_func=self.draft_mode_func,
                prefix_len_list=prefix_len_list,
                args=self.args,
                performance_dict=self.performance_dict,
                max_length=self.max_length,
                past_key_values=past_key_values
            )
            self.performance_dict["model_forward_count"]["verify"] += 1
            accepted = self.performance_dict["total_accept_count"]
            checked  = self.performance_dict["total_checked_count"]
            if checked > 0:
                self.performance_dict["accept_ratio"] = accepted / checked

            wandb.log({
                "accept_ratio_step": self.performance_dict["accept_ratio"],   # ★ step 단위 그래프용
                "draft_fw":  self.performance_dict["model_forward_count"]["draft"],
                "verify_fw": self.performance_dict["model_forward_count"]["verify"],
                # … 이미 있던 다른 로그 필드 …
            })
            
            if self.args.use_cache:
                past_key_values = _verify_out.past_key_values
                fractal_key_values_list = _verify_out.fractal_key_values_list

            for s in active_slots:
                if not s["continue_draft"]:
                    print(">> Generated text:", self.tokenizer.decode(s["input_ids"][0]))
                    s["active"] = False
                    self._fill_slot(s)

            if all(not s["active"] for s in self.slots):
                done = True     
        print("[ParallelSPGenerator] Done all.")
        return self.performance_dict