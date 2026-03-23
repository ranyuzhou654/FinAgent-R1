"""veRL-based multi-turn GRPO training entrypoint for FinAgent-R1."""

from __future__ import annotations

from pathlib import Path

import hydra
import ray
import torch
from omegaconf import OmegaConf

from training.finagent_generation import GenerationConfig, LLMGenerationManager
from training.reward_functions import RewardWeights, compute_finagent_score
from training.search_r1_compat import ensure_search_r1_on_path


ensure_search_r1_on_path()

from verl import DataProto  # noqa: E402
import verl.trainer.ppo.ray_trainer as ray_trainer  # noqa: E402


ray_trainer.LLMGenerationManager = LLMGenerationManager
ray_trainer.GenerationConfig = GenerationConfig

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[1]


class RewardManager:
    def __init__(self, tokenizer, weights: RewardWeights, num_examine: int = 0) -> None:
        self.tokenizer = tokenizer
        self.weights = weights
        self.num_examine = num_examine
        self._printed = 0

    def __call__(self, data: DataProto) -> torch.Tensor:
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for index in range(len(data)):
            item = data[index]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = item.batch["responses"]
            valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequence = torch.cat((valid_prompt_ids, valid_response_ids))
            sequence_text = self.tokenizer.decode(sequence)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            score = compute_finagent_score(sequence_text, ground_truth=ground_truth, weights=self.weights)
            reward_tensor[index, valid_response_length - 1] = score

            if self._printed < self.num_examine:
                self._printed += 1
                print(sequence_text)
                print(f"[reward] {score:.4f}")
        return reward_tensor


def _build_reward_weights(config) -> RewardWeights:
    reward_config = config.finagent_reward
    return RewardWeights(
        accuracy=reward_config.accuracy,
        format=reward_config.format,
        tool_use=reward_config.tool_use,
        multi_tool=reward_config.multi_tool,
        reasoning_after_observation=reward_config.reasoning_after_observation,
        no_tool_penalty=reward_config.no_tool_penalty,
        overuse_penalty=reward_config.overuse_penalty,
        invalid_action_penalty=reward_config.invalid_action_penalty,
        max_tool_calls=reward_config.max_tool_calls,
        tolerance=reward_config.tolerance,
    )


@hydra.main(config_path="../configs", config_name="verl_ppo_finqa", version_base=None)
def main(config) -> None:
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config) -> None:
    from pprint import pprint
    from transformers import AutoTokenizer
    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.actor_rollout_ref.actor.strategy == "fsdp":
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError(f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}")

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    resource_pool_id = "global_pool"
    resource_pool_spec = {
        resource_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: resource_pool_id,
        Role.Critic: resource_pool_id,
        Role.RefPolicy: resource_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(f"Unsupported reward model strategy: {config.reward_model.strategy}")
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = resource_pool_id

    weights = _build_reward_weights(config)
    reward_fn = RewardManager(tokenizer=tokenizer, weights=weights, num_examine=0)
    val_reward_fn = RewardManager(tokenizer=tokenizer, weights=weights, num_examine=1)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping),
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()

