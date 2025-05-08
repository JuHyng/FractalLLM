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
        
        slots[batch_idx]["input_ids"] = slots[batch_idx]["input_ids"] [:, :-slots[batch_idx]["final_draft_len"]]
        for i in range(prefix_len, prefix_len + slots[batch_idx]["final_draft_len"]+1):
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
    past_key_values: Optional[Cache] = None,
    max_length: Optional[int] = None,
):
    """
    ▸ 드래프트 토큰을 검증하고, 맞은/틀린 토큰 수에 따라 슬롯 상태를 업데이트한다.
    ▸ performance_dict에 누적 토큰 기반 가중 평균 accept_ratio를 기록한다.
    """
    # 1) 모델을 verify-mode로 전환
    draft_mode_func(model, is_verify=True)
    performance_dict["model_forward_count"]["verify"] += 1
    start_t = time.time()

    # 2) 이번 스텝에서 active 상태인 슬롯만 추출
    active_idx = [i for i, s in enumerate(slots) if s["continue_draft"]]
    if not active_idx:
        return slots, None, 0

    # 3) 캐시 관련 len 정보 (leftover_len_list) 계산
    if args.use_cache and past_key_values is not None:
        cache_batch_split = past_key_values.batch_split(len(prefix_len_list), 1)
        leftover_len_list = [cbs.get_seq_length() for cbs in cache_batch_split]
    else:
        leftover_len_list = [0] * len(prefix_len_list)

    slots = [slots[i] for i in active_idx]

    # 4) 배치 입력(Tensor) 구성
    batch_input_ids, batch_attention_mask, slot_map, max_len = pad_and_combine_slots(
        slots,
        pad_token_id=tokenizer.pad_token_id,
        leftover_len_list=leftover_len_list,
    )

    target_device = next(model.parameters()).device
    batch_input_ids = batch_input_ids.to(target_device, non_blocking=True)
    batch_attention_mask = batch_attention_mask.to(target_device, non_blocking=True)

    # 5) cache_position (use_cache=True일 때) 계산
    if args.use_cache:
        past_seen_tokens = past_key_values.get_seq_length()
        seq_len_new = batch_input_ids.size(1)
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len_new,
            device=batch_input_ids.device,
        )
    else:
        cache_position = None

    # 6) 모델 forward
    with torch.no_grad():
        outputs = model.forward(
            input_ids=batch_input_ids,
            # attention_mask=batch_attention_mask,  # 필요 시 주석 해제
            use_cache=args.use_cache,
            past_key_values=past_key_values if args.use_cache else None,
            cache_position=cache_position,
        )

    logits = outputs.logits

    # ────────────────────────────────────────────────────────────────
    # 7) Verify 각 슬롯 & 가중 평균 계산용 토큰 카운팅
    # ────────────────────────────────────────────────────────────────
    matched_tokens_step = 0   # 이번 verify-step에서 맞은 토큰 총수
    checked_tokens_step = 0   # 이번 verify-step에서 검사한 토큰 총수
    verified_count_last   = 0 # 마지막 슬롯 verified 토큰 수 (호환성 유지용)

    for i, slot_data in enumerate(slots):
        verified_count = 0
        total_checked = slot_data["final_draft_len"] + 1
        fail_draft = False
        first_fail_id = None

        # (A) 토큰별 검증
        for j in range(total_checked):
            pos = (
                j
                + slot_data["current_input_ids"].shape[1]
                - (total_checked + 1)
            )
            token_logits = logits[i, pos, :]
            verify_pred_id = torch.argmax(token_logits).item()
            curr_id = slot_data["current_input_ids"][0, pos + 1].item()

            if verify_pred_id == curr_id:
                verified_count += 1
            else:
                fail_draft = True
                first_fail_id = verify_pred_id
                break  # mismatch 발견 즉시 중단

        # (B) 가중 합산용 카운터 누적
        matched_tokens_step += verified_count
        checked_tokens_step += total_checked
        verified_count_last = verified_count  # 마지막 슬롯 값을 갱신

        # (C) 슬롯별 accept_ratio(참고용) 저장
        slot_data["accept_ratio_slot"] = verified_count / total_checked

        # (D) 토큰 교정 및 상태 갱신
        if fail_draft and first_fail_id is not None:
            keep = verified_count            # 맞은 토큰 수
            pre_ids = slot_data["input_ids"][:, : -(total_checked - keep)]
            label_tensor = torch.tensor(
                [[first_fail_id]],
                device=pre_ids.device,
                dtype=pre_ids.dtype,
            )
            slot_data["input_ids"] = torch.cat([pre_ids, label_tensor], dim=-1)
            added = keep + 1                 # mismatch 포함
        else:
            added = verified_count           # 모두 일치했으면 그대로

        slot_data["total_new_tokens"] += added
        performance_dict["new_tokens"] += added
        slot_data["continue_draft"] = slot_data["total_new_tokens"] < max_length

        # prompt 길이를 제외한 실제 신규 토큰 수
        slot_data["total_new_tokens"] = (
            slot_data["input_ids"].shape[1] - slot_data["prompt_len"]
        )

    # ────────────────────────────────────────────────────────────────
    # 8) performance_dict 누적 & 가중 평균 accept_ratio 계산
    # ────────────────────────────────────────────────────────────────
    performance_dict.setdefault("matched_tokens", 0)
    performance_dict.setdefault("checked_tokens", 0)

    performance_dict["matched_tokens"] += matched_tokens_step
    performance_dict["checked_tokens"] += checked_tokens_step

    if performance_dict["checked_tokens"]:
        performance_dict["accept_ratio"] = (
            performance_dict["matched_tokens"] / performance_dict["checked_tokens"]
        )
    else:
        performance_dict["accept_ratio"] = 0.0

    verify_time = time.time() - start_t
    performance_dict["verify_time"] += verify_time

    # verified_count_last는 이전 API를 깨뜨리지 않기 위해 그대로 반환
    return slots, outputs, verified_count_last

    
            

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
                if remaining <= 0:                      # 더 못 붙이면 드래프트 중단
                    slot["continue_draft"] = False
                    continue

                final_draft_len = min(self.args.draft_len, remaining)

                # 3️⃣ draft_ids 잘라서 붙이기
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
                past_key_values=past_key_values
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
        
        