"""Tensor helpers used by the FinAgent veRL generation loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int


class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: dict[str, torch.Tensor], keys: list[str], cut_left: bool = True) -> dict[str, torch.Tensor]:
        effective_len = tensor_dict["attention_mask"].sum(dim=1).max()
        result = tensor_dict.copy()
        for key in keys:
            result[key] = tensor_dict[key][:, -effective_len:] if cut_left else tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: list[torch.Tensor], pad_to_left: bool = True) -> torch.Tensor:
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def example_level_pad(
        self,
        responses: torch.Tensor,
        responses_str: list[str],
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[str]]:
        assert int(active_mask.sum().item()) == responses.shape[0]
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len),
            self.config.pad_token_id,
            dtype=responses.dtype,
            device=responses.device,
        )
        padded_responses[active_mask] = responses

        padded_response_strs = [""] * batch_size
        source_index = 0
        for index, is_active in enumerate(active_mask.tolist()):
            if is_active:
                padded_response_strs[index] = responses_str[source_index]
                source_index += 1
        return padded_responses, padded_response_strs

