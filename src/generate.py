# src/generate.py

import time
from copy import deepcopy
from collections import deque
from typing import List, Optional

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from src.model import FractalLlamaForCausalLM
from src.utils import FlopsCounter


def pad_and_combine_slots(
    slots: List[dict],
    pad_token_id: int,
    leftover_len_list: Optional[List[int]] = None,
) -> (torch.Tensor, torch.Tensor, List[int], int): # type: ignore
    """
    여러 슬롯의 `current_input_ids`를 같은 길이(max)로 패딩하고
    attention mask를 생성합니다.
    Returns (batch_input_ids, batch_attention_mask, slot_map, max_len).
    """
    leftover_len_list = leftover_len_list or [0] * len(slots)

    # 각 슬롯의 현재 길이
    lengths = [s["current_input_ids"].size(1) for s in slots]
    max_len = max(lengths, default=0)

    batch_ids, batch_masks, slot_map = [], [], []
    for idx, slot in enumerate(slots):
        ids = slot["current_input_ids"]
        total_len = ids.size(1)
        leftover = min(leftover_len_list[idx], total_len)

        # (A) pad to max_len
        pad_len = max_len - total_len
        if pad_len > 0:
            pad = torch.full((1, pad_len), pad_token_id, dtype=ids.dtype, device=ids.device)
            ids = torch.cat([ids, pad], dim=1)

        # (B) attention mask: 새로운 토큰은 1, 나머지는 0
        mask = torch.zeros((1, total_len), dtype=torch.long, device=ids.device)
        mask[:, leftover:] = 1
        if pad_len > 0:
            pad_mask = torch.zeros((1, pad_len), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, pad_mask], dim=1)

        batch_ids.append(ids)
        batch_masks.append(mask)
        slot_map.append(idx)

    if not batch_ids:
        return torch.zeros(0, 0), torch.zeros(0, 0), [], 0

    return (
        torch.cat(batch_ids, dim=0),
        torch.cat(batch_masks, dim=0),
        slot_map,
        max_len,
    )


def draft_forward(
    args,
    slots: List[dict],
    model: FractalLlamaForCausalLM,
    tokenizer,
    draft_mode_func,
    performance_dict: dict,
    prefix_len_list: List[int],
    past_key_values: Optional[DynamicCache] = None,
    fractal_key_values_list: Optional[List[DynamicCache]] = None,
) -> (List[dict], object): # type: ignore
    """
    ① 모델을 드래프트 모드로 전환  
    ② pad_and_combine_slots 실행  
    ③ FractalLlamaForCausalLM.forward 호출하여 logits 얻기  
    ④ 각 슬롯별로 토큰 예측 후 `input_ids`에 추가
    """
    draft_mode_func(model, is_verify=False)
    performance_dict["model_forward_count"]["draft"] += 1

    # leftover 길이 추출
    if args.use_cache and past_key_values is not None:
        splits = past_key_values.batch_split(len(prefix_len_list), 1)
        leftover = [c.get_seq_length() for c in splits]
    else:
        leftover = [0] * len(prefix_len_list)

    batch_ids, _, slot_map, _ = pad_and_combine_slots(
        slots, tokenizer.pad_token_id, leftover
    )

    # cache 위치 계산
    cache_pos = None
    if args.use_cache and past_key_values is not None:
        seen = past_key_values.get_seq_length()
        new_len = batch_ids.size(1)
        cache_pos = torch.arange(seen, seen + new_len, device=batch_ids.device)

    # forward
    with torch.no_grad():
        outputs = model(
            input_ids=batch_ids,
            use_cache=args.use_cache,
            past_key_values=past_key_values if args.use_cache else None,
            fractal_key_values_list=fractal_key_values_list if args.use_cache else None,
            prefix_len_list=prefix_len_list,
            cache_position=cache_pos,
        )
    logits = outputs.logits

    # 슬롯별 token append
    for i, slot in enumerate(slots):
        prefix = prefix_len_list[i]
        preds = logits[i, prefix : prefix + args.draft_len + 1].argmax(dim=-1)
        slot["input_ids"] = torch.cat(
            [slot["input_ids"][:, :-args.draft_len], preds.unsqueeze(0)], dim=1
        )

    return slots, outputs


def verify_forward(
    args,
    slots: List[dict],
    model: FractalLlamaForCausalLM,
    tokenizer,
    draft_mode_func,
    prefix_len_list: List[int],
    performance_dict: dict,
    past_key_values: Optional[DynamicCache] = None,
    max_length: Optional[int] = None,
) -> (List[dict], object, int): # type: ignore

    draft_mode_func(model, is_verify=True)
    performance_dict["model_forward_count"]["verify"] += 1

    # 검증할 슬롯 인덱스
    active = [s for s in slots if s["continue_draft"]]
    if not active:
        return slots, None, 0

    # pad & combine
    leftover = [0] * len(prefix_len_list)
    if args.use_cache and past_key_values is not None:
        splits = past_key_values.batch_split(len(prefix_len_list), 1)
        leftover = [c.get_seq_length() for c in splits]

    batch_ids, _, _, _ = pad_and_combine_slots(
        active, tokenizer.pad_token_id, leftover
    )

    # cache pos
    cache_pos = None
    if args.use_cache and past_key_values is not None:
        seen = past_key_values.get_seq_length()
        new_len = batch_ids.size(1)
        cache_pos = torch.arange(seen, seen + new_len, device=batch_ids.device)

    with torch.no_grad():
        outputs = model(
            input_ids=batch_ids,
            use_cache=args.use_cache,
            past_key_values=past_key_values if args.use_cache else None,
            cache_position=cache_pos,
        )
    logits = outputs.logits

    verified = 0
    for idx, slot in enumerate(active):
        # 검증 가능한 토큰 개수
        checks = args.draft_len + 1
        match = 0
        for p in range(checks):
            pos = slot["current_input_ids"].size(1) - checks + p
            pred = logits[idx, pos].argmax().item()
            actual = slot["current_input_ids"][0, pos + 1].item()
            if pred == actual:
                if args.print_draft:
                    print(f"[MATCH] Draft: {pred}, Actual: {actual}")
                
                match += 1
            else:
                print(f"[MISMATCH] Draft: {pred}, Actual: {actual}")
                # 실패 시 첫 mismatch 이후 correct로 대체
                replacement = torch.tensor([[pred]], device=batch_ids.device)
                slot["input_ids"] = torch.cat(
                    [slot["input_ids"][:, : pos + 1], replacement], dim=1
                )
                slot["total_new_tokens"] += (match + 1)
                break

        slot["accept_ratio"] = match / checks
        performance_dict["accept_ratio"] = slot["accept_ratio"]
        performance_dict["new_tokens"] += match
        verified = match

        # 최대 길이 도달 시 드래프트 중단
        if slot["total_new_tokens"] >= max_length:
            slot["continue_draft"] = False

    return slots, outputs, verified


class ParallelSPGenerator(nn.Module):
    """
    Speculative Parallel Generation:
    Draft 모드 → Verify 모드를 batch-wise로 반복 실행합니다.
    """
    def __init__(
        self,
        model: FractalLlamaForCausalLM,
        tokenizer,
        draft_mode_func,
        args,
        data_queue: deque,
        draft_tokens: List[int],
        performance_dict: dict,
        max_length: int,
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

        # 슬롯 초기화
        self.slots = [
            {
                "input_ids": None,
                "current_input_ids": None,
                "continue_draft": True,
                "total_new_tokens": 0,
            }
            for _ in range(args.batch_size)
        ]

    def _fill_slot(self, slot):
        if not self.data_queue:
            slot["continue_draft"] = False
            return
        q, _ = self.data_queue.popleft()
        inp = self.tokenizer(q, return_tensors="pt")
        slot["input_ids"] = inp["input_ids"]
        slot["current_input_ids"] = inp["input_ids"]

    def forward(self) -> dict:
        # 초기 슬롯 채우기
        for slot in self.slots:
            self._fill_slot(slot)

        past, fractals = None, None
        if self.args.use_cache:
            past = DynamicCache()
            fractals = [DynamicCache() for _ in self.args.draft_layer_indexes]

        while any(s["continue_draft"] for s in self.slots):
            active = [s for s in self.slots if s["continue_draft"]]
            prefixes = [s["input_ids"].size(1) for s in active]

            # draft 단계
            for s in active:
                draft_ids = torch.tensor([self.draft_tokens], device=s["input_ids"].device)
                s["input_ids"] = torch.cat([s["input_ids"], draft_ids], dim=1)

            active, past = draft_forward(
                self.args, active, self.model, self.tokenizer,
                self.draft_mode_func, self.performance_dict,
                prefixes, past, fractals,
            )

            # verify 단계
            active, past, _ = verify_forward(
                self.args, active, self.model, self.tokenizer,
                self.draft_mode_func, prefixes,
                self.performance_dict, past, self.max_length
            )

            # 새로운 슬롯 채우기
            for s in self.slots:
                if not s["continue_draft"]:
                    self._fill_slot(s)

        return self.performance_dict
