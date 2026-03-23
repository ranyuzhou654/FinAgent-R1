"""Search-R1-style multi-turn generation loop adapted for FinAgent tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import requests
import torch

from training.search_r1_compat import ensure_search_r1_on_path
from training.tensor_helper import TensorConfig, TensorHelper
from tools.calculator_tool import execute_calculate
from tools.search_tool import get_search_tool
from tools.sql_tool import execute_sql


ensure_search_r1_on_path()

from verl import DataProto  # noqa: E402


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str | None = None
    topk: int = 3
    observation_tag: str = "observation"


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            )
        )

    def _batch_tokenize(self, responses: list[str]) -> torch.Tensor:
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def _truncate_to_first_action_completion(self, text: str) -> str:
        candidates = []
        for closing_tag in ("</search>", "</calculate>", "</sql>", "</answer>"):
            position = text.find(closing_tag)
            if position != -1:
                candidates.append((position + len(closing_tag), closing_tag))
        if not candidates:
            return text
        earliest_end = min(candidates, key=lambda item: item[0])[0]
        return text[:earliest_end]

    def _postprocess_responses(self, responses: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        responses_str = [self._truncate_to_first_action_completion(text) for text in responses_str]
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: list[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, : self.config.max_obs_length]
        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> DataProto:
        new_input_ids = self.tensor_fn.concatenate_with_padding(
            [rollings.batch["input_ids"], cur_responses, next_obs_ids]
        )
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, int(effective_len.item()))
        new_rollings = DataProto.from_dict(
            {
                "input_ids": new_input_ids[:, -max_len:],
                "position_ids": new_position_ids[:, -max_len:],
                "attention_mask": new_attention_mask[:, -max_len:],
            }
        )
        new_rollings.meta_info.update(rollings.meta_info)
        return new_rollings

    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor | None = None,
        pad_to_left: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return concatenated.gather(1, sorted_indices), concatenated_with_info.gather(1, sorted_indices)

    def _update_right_side(
        self,
        right_side: dict[str, torch.Tensor],
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
            right_side["responses"],
            right_side["responses_with_info_mask"],
            cur_responses,
            next_obs_ids,
            pad_to_left=False,
        )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, int(effective_len.item()))
        return {
            "responses": responses[:, :max_len],
            "responses_with_info_mask": responses_with_info_mask[:, :max_len],
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch["input_ids"].shape[0]
        remainder = batch_size % num_gpus
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        padding_size = num_gpus - remainder
        padded_batch = {}
        for key, value in active_batch.batch.items():
            pad_sequence = value[0:1].repeat(padding_size, *[1] * (len(value.shape) - 1))
            padded_batch[key] = torch.cat([value, pad_sequence], dim=0)
        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        trimmed_batch = {key: value[:-padding_size] for key, value in padded_output.batch.items()}
        if hasattr(padded_output, "meta_info") and padded_output.meta_info:
            trimmed_meta = {}
            for key, value in padded_output.meta_info.items():
                if isinstance(value, torch.Tensor):
                    trimmed_meta[key] = value[:-padding_size]
                else:
                    trimmed_meta[key] = value
            padded_output.meta_info = trimmed_meta
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> DataProto:
        original_left_side = {"input_ids": initial_input_ids[:, -self.config.max_start_length :]}
        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }
        active_mask = torch.ones(gen_batch.batch["input_ids"].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        tool_stats = torch.zeros(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        rollings = gen_batch

        for _ in range(self.config.max_turns):
            if not bool(active_mask.sum()):
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
            )
            rollings_active = DataProto.from_dict({key: value[active_mask] for key, value in rollings.batch.items()})
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch["responses"])
            responses_ids, responses_str = self.tensor_fn.example_level_pad(responses_ids, responses_str, active_mask)
            next_obs, dones, valid_action, tool_used = self.execute_predictions(responses_str, active_mask)
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            tool_stats += torch.tensor(tool_used, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)

        if bool(active_mask.sum()):
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
            )
            rollings_active = DataProto.from_dict({key: value[active_mask] for key, value in rollings.batch.items()})
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch["responses"])
            responses_ids, responses_str = self.tensor_fn.example_level_pad(responses_ids, responses_str, active_mask)
            _, dones, valid_action, tool_used = self.execute_predictions(responses_str, active_mask, do_tools=False)
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            tool_stats += torch.tensor(tool_used, dtype=torch.int)
            original_right_side = self._update_right_side(original_right_side, responses_ids)

        meta_info["turns_stats"] = turns_stats.tolist()
        meta_info["active_mask"] = active_mask.tolist()
        meta_info["valid_action_stats"] = valid_action_stats.tolist()
        meta_info["tool_stats"] = tool_stats.tolist()
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(
        self,
        left_side: dict[str, torch.Tensor],
        right_side: dict[str, torch.Tensor],
        meta_info: dict[str, Any],
    ) -> DataProto:
        final_output = right_side.copy()
        final_output["prompts"] = left_side["input_ids"]
        final_output["input_ids"] = torch.cat([left_side["input_ids"], right_side["responses"]], dim=1)
        final_output["attention_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(final_output["responses"]),
            ],
            dim=1,
        )
        final_output["info_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(final_output["responses_with_info_mask"]),
            ],
            dim=1,
        )
        final_output["position_ids"] = self.tensor_fn.create_position_ids(final_output["attention_mask"])
        output = DataProto.from_dict(final_output)
        output.meta_info.update(meta_info)
        return output

    def _parse_action(self, prediction: str) -> tuple[str | None, str]:
        pattern = r"<(search|calculate|sql|answer)>(.*?)</\1>"
        match = re.search(pattern, prediction, re.DOTALL)
        if not match:
            return None, ""
        return match.group(1), match.group(2).strip()

    def execute_predictions(
        self,
        predictions: list[str],
        active_mask: torch.Tensor,
        do_tools: bool = True,
    ) -> tuple[list[str], list[int], list[int], list[int]]:
        parsed = [self._parse_action(prediction) for prediction in predictions]
        search_queries = [content for action, content in parsed if action == "search"]
        if do_tools:
            search_results = self.batch_search(search_queries)
        else:
            search_results = [""] * len(search_queries)

        next_obs: list[str] = []
        dones: list[int] = []
        valid_action: list[int] = []
        tool_used: list[int] = []

        for (action, content), active in zip(parsed, active_mask.tolist()):
            if not active:
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                tool_used.append(0)
                continue

            if action == "answer":
                next_obs.append("")
                dones.append(1)
                valid_action.append(1)
                tool_used.append(0)
            elif action == "search":
                next_obs.append(self._format_observation(search_results.pop(0)))
                dones.append(0)
                valid_action.append(1)
                tool_used.append(1)
            elif action == "calculate":
                next_obs.append(self._format_observation(execute_calculate(content)))
                dones.append(0)
                valid_action.append(1)
                tool_used.append(1)
            elif action == "sql":
                next_obs.append(self._format_observation(execute_sql(content)))
                dones.append(0)
                valid_action.append(1)
                tool_used.append(1)
            else:
                next_obs.append(
                    self._format_observation(
                        "Invalid action. Use one of <search>...</search>, <calculate>...</calculate>, "
                        "<sql>...</sql>, or <answer>...</answer>."
                    )
                )
                dones.append(0)
                valid_action.append(0)
                tool_used.append(0)

        assert not search_results
        return next_obs, dones, valid_action, tool_used

    def _format_observation(self, content: str) -> str:
        return f"\n\n<{self.config.observation_tag}>{content.strip()}</{self.config.observation_tag}>\n\n"

    def batch_search(self, queries: list[str]) -> list[str]:
        if not queries:
            return []
        raw_result = self._batch_search(queries).get("result", [])
        return [self._passages_to_string(item) for item in raw_result]

    def _batch_search(self, queries: list[str]) -> dict[str, Any]:
        if self.config.search_url:
            payload = {
                "queries": queries,
                "topk": self.config.topk,
                "return_scores": True,
                "method": "hybrid",
            }
            response = requests.post(self.config.search_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()

        local_results = get_search_tool().batch_search(queries, method="hybrid", topk=self.config.topk, return_scores=True)
        return {"result": local_results}

    def _passages_to_string(self, retrieval_result: list[dict[str, Any]]) -> str:
        passages = []
        for index, item in enumerate(retrieval_result, start=1):
            document = item["document"] if "document" in item else item
            title = document.get("title", f"doc-{index}")
            contents = document.get("contents", document.get("text", ""))
            passages.append(f"Doc {index} (Title: {title}) {contents}")
        return "\n".join(passages)

