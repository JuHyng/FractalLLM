import torch
import torch.nn as nn
import time
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
    
    performance_dict["model_forward_count"]["draft"] += 1
    
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
    
    start_t = time.time()

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
            # attention_mask=batch_attention_mask,
            use_cache=args.use_cache,
            past_key_values=past_key_values if args.use_cache else None,
            fractal_key_values_list=fractal_key_values_list if args.use_cache else None,
            prefix_len_list=prefix_len_list,
            cache_position=cache_position if args.use_cache else None,
        )
    logits = outputs.logits
    
    if args.use_cache:
        cache_batch_split = past_key_values.batch_split(len(prefix_len_list), 1)
    
    if args.use_cache:
        print("[After Draft Before Verify]past_key_values._seen_tokens", past_key_values._seen_tokens)
        print("[After Draft Before Verify]fractal_key_values_list[0]._seen_tokens", fractal_key_values_list[0]._seen_tokens)
            
    for batch_idx in range(len(prefix_len_list)):
        prefix_len = prefix_len_list[batch_idx]
        
        if args.use_cache:
            if slots[batch_idx]["total_new_tokens"] > 0:
                prefix_len = 1
            cache_batch_split[batch_idx].crop(-(args.draft_len+1))
            past_key_values = DynamicCache.from_batch_splits(cache_batch_split)
        
        slots[batch_idx]["input_ids"] = slots[batch_idx]["input_ids"] [:, :-args.draft_len]
        for i in range(prefix_len, prefix_len + args.draft_len+1):
            token_logits = logits[batch_idx, i-1, :]
            token_prob = token_logits
            token_pred = torch.argmax(token_prob).item()
            slots[batch_idx]["input_ids"] = torch.cat([slots[batch_idx]["input_ids"], torch.tensor([[token_pred]], device=slots[batch_idx]["input_ids"].device)], dim=-1)   
            
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
    past_key_values:Optional[Cache]=None,
    max_length:Optional[int]=None,
    first_wrong_flag:Optional[bool]=True,
):
    draft_mode_func(model, is_verify=True)
    
    performance_dict["model_forward_count"]["verify"] += 1
    start_t = time.time()
    
    active_idx = [i for i, s in enumerate(slots) if s["continue_draft"]]
    if not active_idx:
        return slots, None, 0
    
    if args.use_cache and past_key_values is not None:
        cache_batch_split = past_key_values.batch_split(len(prefix_len_list), 1)
        leftover_len_list = [cbs.get_seq_length() for cbs in cache_batch_split]
    else:
        leftover_len_list = [0]*len(prefix_len_list)
        
    slots = [slots[i] for i in active_idx]

    batch_input_ids, batch_attention_mask, slot_map, max_len = pad_and_combine_slots(
        slots,
        pad_token_id=tokenizer.pad_token_id,
        leftover_len_list=leftover_len_list
    )
    
    
    if args.use_cache:
        past_seen_tokens = past_key_values.get_seq_length()
        seq_len_new = batch_input_ids.size(1)     
        
        print("[Draft] past_seen_tokens:", past_seen_tokens)
        print("[Draft] seq_len_new:", seq_len_new)
        
        cache_position = torch.arange(
            past_seen_tokens, 
            past_seen_tokens + seq_len_new, 
            device=batch_input_ids.device
        )
        print("[Draft] cache_position:", cache_position)
    
    with torch.no_grad():
        outputs = model.forward(
            input_ids=batch_input_ids,
            # attention_mask=batch_attention_mask,
            use_cache=args.use_cache,
            past_key_values=past_key_values if args.use_cache else None,
            cache_position=cache_position if args.use_cache else None,
        )
        
    logits = outputs.logits
    
    for i, slot_data in enumerate(slots):        
        verified_count = 0
        total_checked = args.draft_len + 1
        fail_draft = False
        first_fail_label = None

        for pos in range(args.draft_len+1):
            pos = pos + slot_data["current_input_ids"].shape[1] - (args.draft_len+2)
            token_logits = logits[i, pos, :]
            verify_pred_id = torch.argmax(token_logits).item()
            curr_id = slot_data["current_input_ids"][0, pos+1].item()
            
            if args.print_draft:
                pred_str = tokenizer.decode(verify_pred_id)
                curr_str = tokenizer.decode(curr_id)

            if verify_pred_id == curr_id:
                if first_wrong_flag:
                    first_wrong_flag = False
                
                if args.print_draft:
                    print(f"[Verify] MATCH: slot {i} at position {pos}: {pred_str} (current: {curr_str})")
                verified_count += 1
            else:
                if first_wrong_flag:
                    input("[Verify] FIRST MISMATCH: slot {i} at position {pos}: {pred_str} (current: {curr_str})")
                
                if args.print_draft:
                    print(f"[Verify] MISMATCH: slot {i} position {pos}: {pred_str} (current: {curr_str})")
                
                fail_draft = True
                first_fail_label = verify_pred_id
                break

        accept_ratio = float(verified_count) / float(total_checked)
        slot_data["accept_ratio"] = accept_ratio
        performance_dict["accept_ratio"] = accept_ratio

        if fail_draft and first_fail_label is not None:
            pre_ids = slot_data["input_ids"][:, :-(args.draft_len+1 - verified_count)]
            label_tensor = torch.tensor([[first_fail_label]], 
                                device=pre_ids.device, dtype=pre_ids.dtype)
            slot_data["input_ids"] = torch.cat([pre_ids, label_tensor], dim=-1)
            slot_data["total_new_tokens"] += (verified_count+1) # +1: mismatch token
            slot_data["continue_draft"] = True 
            performance_dict["new_tokens"]+= (verified_count+1)
            
        else:
            slot_data["continue_draft"] = True  
            slot_data["total_new_tokens"] += verified_count
            performance_dict["new_tokens"] += verified_count
            
        if slot_data["total_new_tokens"] >= max_length:
            slot_data["continue_draft"] = False
                
        verify_time = time.time() - start_t
        performance_dict["verify_time"] += verify_time

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
        
        # DEBUG
        first_wrong_flag = True
        
        while not done:
            active_slots = [s for s in self.slots if s["active"]]
            if len(active_slots) == 0:
                done = True
                break              
            
            prefix_len_list = []
            for slot in active_slots:
                prefix_len_list.append(slot["input_ids"].shape[1])
                
            for slot in active_slots:
                slot["draft_iteration"] += 1
                draft_t = torch.tensor([draft_ids], device=slot["input_ids"].device)
                slot["input_ids"] = torch.cat([slot["input_ids"], draft_t], dim=-1)
                
            
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
            
            # DEBUG
            if self.args.print_draft:
                print("[Before Draft] decoded text:", self.tokenizer.decode(active_slots[i]["input_ids"][0]))

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
            
            if self.args.use_cache:
                past_key_values = _draft_out.past_key_values
                fractal_key_values_list = _draft_out.fractal_key_values_list
                
                if past_key_values.get_seq_length() > 0:
                    cache_batch_split = past_key_values.batch_split(len(active_slots), 1)
                    for i in range(len(cache_batch_split)):
                        if cache_batch_split[i].get_seq_length() > 0:
                            active_slots[i]["current_input_ids"] = active_slots[i]["input_ids"][:, -(1+self.args.draft_len+1):] # 1(last token of prefix) + draft_tokens (draft_len+1)
                            cache_batch_split[i].crop(-(1+self.args.draft_len+1))
                        else:
                            active_slots[i]["current_input_ids"] = active_slots[i]["input_ids"]
                        
                    past_key_values = DynamicCache.from_batch_splits(cache_batch_split)
                    print("[After Draft Before Verify] past_key_values.get_seq_length()", past_key_values.get_seq_length())
                    
                for fractal_key_values in fractal_key_values_list:
                    if fractal_key_values.get_seq_length() > 0:
                        print("[After Draft Before Verify] fractal_key_values.__seen_tokens", fractal_key_values._seen_tokens)
                        fractal_key_values.crop(-(1+self.args.draft_len+1))
                        print("[After Draft Before Verify] fractal_key_values.get_seq_length()", fractal_key_values.get_seq_length())
                    
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
                past_key_values=past_key_values,
                first_wrong_flag=first_wrong_flag,
            )
            
            if self.args.use_cache:
                past_key_values = _verify_out.past_key_values
                
                if past_key_values.get_seq_length() > 0:
                    slot_data = active_slots[0]  # 혹은 self.slots[0] 등
                    fractal_cache = fractal_key_values_list[0]  # 프랙탈 캐시도 하나만 있다고 가정

                    # (A) 길이 확인
                    fractal_prefix_len = fractal_cache.get_seq_length()  # 프랙탈 캐시가 본 prefix 길이
                    main_prefix_len    = past_key_values.get_seq_length() # 메인 캐시 길이

                    # "아직 프랙탈이 안 본 길이" => new_len
                    new_len = main_prefix_len - fractal_prefix_len
                    if new_len < 0:
                        new_len = 0

                    # (B) 현재 slot의 input_ids (shape: (1, total_seq_len))
                    input_ids_2d = slot_data["input_ids"]
                    total_seq_len = input_ids_2d.shape[1]

                    # clamp
                    if new_len > total_seq_len:
                        new_len = total_seq_len

                    # (C) new_len > 0 이면, 그만큼 슬라이싱
                    if new_len > 0:
                        # 맨 끝에서 new_len개의 토큰을 떼온다
                        needed_input = input_ids_2d[:, -new_len:]  # shape (1, new_len)
                    else:
                        # 갱신할 부분 없으면 빈 텐서
                        needed_input = torch.zeros(
                            (1,0),
                            dtype=input_ids_2d.dtype,
                            device=input_ids_2d.device
                        )

                    # ------------------------------------------------
                    # 3) 프랙탈 캐시만 보충 계산
                    #    (메인 캐시는 이미 완료)
                    # ------------------------------------------------
                    if new_len > 0:
                        fractal_key_values_list = self.model.compute_fractal_key_values(
                            input_ids=needed_input,           # (1, new_len)짜리
                            attention_mask=None,             # 필요하다면 직접 생성. 여기서는 None->함수 내부에서 처리
                            past_key_values=past_key_values, # 메인 캐시 (길이=main_prefix_len)
                            fractal_key_values_list=fractal_key_values_list,
                            use_cache=True
                        )
                    print("[After Verify] past_key_values.get_seq_length()", past_key_values.get_seq_length())
                    print("[After Verify] fractal_key_values_list[0].get_seq_length()", fractal_key_values_list[0].get_seq_length())
                    input()

            for s in active_slots:
                if not s["continue_draft"]:
                    s["active"] = False
                    self._fill_slot(s)

            if all(not s["active"] for s in self.slots):
                done = True

        print("[ParallelSPGenerator] Done all.")
        return self.performance_dict
        
        
