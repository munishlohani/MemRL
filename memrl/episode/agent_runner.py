from __future__ import annotations

from pathlib import Path
import copy
import logging
from uuid import uuid4
import time
import numpy as np
import json
import random

from .base import BaseEpisodeRunner
from memrl.agent import prompts as agent_prompts
from memrl.agent.base import AgentDecision, BaseAgent, EnvActionDecision, SkillInvocationDecision
from memrl.agent.history import EpisodeHistory
from memrl.configs.config import MempConfig
from memrl.service.memory_service import MemoryService
from typing import Any, Dict, List, Optional
from memrl.service.sleep_consolidation.checkpoint import SleepConsolidationCheckpoint
from memrl.service.formation_judger import (
    TacticalFormationCandidate,
    TacticalFormationJudge,
    TacticalSummaryWriter,
)
from memrl.skills.memory_retrieval import MemoryRetrievalResult, MemoryRetrievalSkill
from memrl.providers.base import BaseLLM
from memrl.utils.q_utils import (
    apply_q_update,
    compute_td_error,
    get_q_omega_salience,
    get_q_salience,
)
from .env_adapter import EpisodeEnvAdapter

MAX_RETRIES=4
RETRY_DELAY=2
MAX_SKILL_INVOCATIONS=3


logger=logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[assignment]

class EpisodeRunner(BaseEpisodeRunner):

    def __init__(
        self,
        *,
        agent: BaseAgent,
        memory_service: MemoryService,
        sleep_checkpoint: Optional[SleepConsolidationCheckpoint],
        env_adapter: EpisodeEnvAdapter,
        config: str,
        output_dir: Path,
        experiment_name: str,
        mode: str = "train",
        run_id: Optional[str] = None,
        retrieve_k: int = 1,
        batch_size: int = 1,
        max_steps: int = 1,
        llm_provider: Optional[BaseLLM] = None,
        strategic_k: int = 3,
        ckpt_resume_enabled: bool = False,
        ckpt_resume_path: Optional[str] = None,
        ckpt_resume_epoch: Optional[int] = None,
        tensorboard_log_dir: Optional[str] = None,
    ):
        self.agent = agent
        self.memory_service = memory_service
        self.llm_provider = llm_provider
        self.env_adapter = env_adapter

        self.config_path = str(config)
        self.config = MempConfig.from_yaml(self.config_path)
        self.memory_config = self.config.memory
        self.experiment_config = self.config.experiment
        self.rl_config = self.config.rl_config

        if sleep_checkpoint is not None:
            self.sleep_checkpoint = sleep_checkpoint
        elif llm_provider is not None:
            self.sleep_checkpoint = SleepConsolidationCheckpoint(
                memory_service=memory_service,
                llm_provider=llm_provider,
                memory_config=self.memory_config,
            )
        else:
            self.sleep_checkpoint = None
        self.formation_judge = (
            TacticalFormationJudge(llm_provider) if llm_provider is not None else None
        )
        self.tactical_summary_writer = (
            TacticalSummaryWriter(llm_provider) if llm_provider is not None else None
        )
        self.memory_retrieval_skill = MemoryRetrievalSkill(
            memory_service=memory_service,
            llm_provider=llm_provider,
            retrieve_k=retrieve_k,
            rl_config=self.rl_config,
        )
        if hasattr(self.agent, "memory_retrieval_skill"):
            try:
                setattr(self.agent, "memory_retrieval_skill", self.memory_retrieval_skill)
            except Exception:
                logger.debug("Failed to attach memory retrieval skill to agent", exc_info=True)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = str(experiment_name)
        self.mode = str(mode)
        self.run_id = str(run_id or time.strftime("%Y%m%d-%H%M%S"))
        self.batch_size = max(1, int(batch_size))
        self.max_steps = max(1, int(max_steps))
        self.retrieve_k = max(1, int(retrieve_k))
        self.strategic_k = max(1, int(strategic_k))

        self.run_dir = self.output_dir / "episode" / f"exp_{self.experiment_name}_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.local_cache_dir = self.run_dir / "local_cache"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_resume_enabled = bool(ckpt_resume_enabled)
        self.ckpt_resume_path = ckpt_resume_path
        self.ckpt_resume_epoch = ckpt_resume_epoch
        self.tensorboard_log_dir = tensorboard_log_dir
        self.tensorboard_writer = self._init_tensorboard_writer(tensorboard_log_dir)

        self.current_step = 0
        self.results_log: List[Dict[str, Any]] = []
        self.episode_histories = [EpisodeHistory() for _ in range(self.batch_size)]
        self.pending_formations: List[Dict[str, Any]] = []
        self.episode_rewards: List[float] = []
        self.active_strategic_node_ids: List[Optional[str]] = []
        self.active_strategic_node_summaries: List[Optional[str]] = []
        self.sleep_bootstrap_tactical_min = getattr(self.memory_config, "n_sleep", None)
        self.metrics_namespace = f"episode/{self.experiment_name}"
        self.metrics_history: List[Dict[str, Any]] = []
        self.metrics_reporter = self._report_metrics

        self.random_seed = getattr(self.experiment_config, "random_seed", None)
        if self.random_seed is not None:
            random.seed(int(self.random_seed))
            np.random.seed(int(self.random_seed))

    def run(self) -> Dict[str, Any]:
        reset_result = self.env_adapter.reset()
        observations = list(reset_result.observations)
        infos = [info if isinstance(info, dict) else {} for info in list(reset_result.infos or [])]
        if not observations:
            raise ValueError("env_adapter.reset() returned no observations")
        if len(infos) < len(observations):
            infos.extend({} for _ in range(len(observations) - len(infos)))

        batch_size = len(observations)
        self.batch_size = batch_size
        self.episode_histories = [EpisodeHistory() for _ in range(batch_size)]
        self.pending_formations = []
        self.episode_rewards = [0.0 for _ in range(batch_size)]
        reward_histories = [[] for _ in range(batch_size)]
        episode_infos = [dict(info) for info in infos]
        done_flags = [False for _ in range(batch_size)]
        step_counts = [0 for _ in range(batch_size)]
        success_flags = [False for _ in range(batch_size)]

        task_descriptions = [
            self._infer_task_description(observations[i], infos[i]) for i in range(batch_size)
        ]
        task_types = [self._infer_task_type(i, infos[i]) for i in range(batch_size)]
        episode_ids = [self._infer_episode_id(i, infos[i]) for i in range(batch_size)]

        active_strategic_node_ids: List[Optional[str]] = [None for _ in range(batch_size)]
        strategic_selection_summaries: List[Optional[str]] = [None for _ in range(batch_size)]
        has_strategic_scaffolds = self._has_strategic_scaffolds()
        for slot_idx in range(batch_size):
            selected_id: Optional[str] = None
            selected_summary: Optional[str] = None
            if has_strategic_scaffolds:
                selected_id, selected_summary = self._select_strategic_scaffold(
                    task_description=task_descriptions[slot_idx],
                    task_type=task_types[slot_idx],
                    observation=str(observations[slot_idx] or ""),
                    history_messages=[],
                    episode_id=episode_ids[slot_idx],
                )
                if selected_id is None:
                    selected_id = self._resolve_strategic_node_id(episode_infos[slot_idx])

            active_strategic_node_ids[slot_idx] = selected_id
            strategic_selection_summaries[slot_idx] = selected_summary
            if selected_id is not None:
                episode_infos[slot_idx]["active_strategic_node_id"] = selected_id
                if selected_summary:
                    episode_infos[slot_idx]["active_strategic_node_summary"] = selected_summary
                self.episode_histories[slot_idx].append_message(
                    {
                        "role": "system",
                        "content": (
                            f"Active strategic scaffold: {selected_id}"
                            + (
                                f"\nSummary: {selected_summary}"
                                if selected_summary
                                else ""
                            )
                        ),
                    }
                )
            elif has_strategic_scaffolds:
                self.episode_histories[slot_idx].append_message(
                    {
                        "role": "system",
                        "content": "No strategic scaffold selected for this episode.",
                    }
                )
            else:
                self.episode_histories[slot_idx].append_message(
                    {
                        "role": "system",
                        "content": "Strategic bootstrap mode: using tactical memories only.",
                    }
                )
        self.active_strategic_node_ids = active_strategic_node_ids
        self.active_strategic_node_summaries = strategic_selection_summaries

        try:
            for step_idx in range(self.max_steps):
                active_slots = [idx for idx, done in enumerate(done_flags) if not done]
                if not active_slots:
                    break

                actions = ["look"] * batch_size
                slot_contexts: List[Dict[str, Any]] = [{} for _ in range(batch_size)]
                for slot_idx in active_slots:
                    history = self.episode_histories[slot_idx]
                    history_messages = self._history_to_messages(history)
                    current_observation = str(observations[slot_idx] or "")
                    active_strategic_node_id = active_strategic_node_ids[slot_idx]
                    if active_strategic_node_id is None and has_strategic_scaffolds:
                        active_strategic_node_id = self._resolve_strategic_node_id(
                            episode_infos[slot_idx] if slot_idx < len(episode_infos) else {}
                        )
                        active_strategic_node_ids[slot_idx] = active_strategic_node_id
                    self.agent.reset(
                        task_description=task_descriptions[slot_idx],
                        task_type=task_types[slot_idx],
                        episode_id=episode_ids[slot_idx],
                    )
                    action, retrieval_result = self._resolve_agent_turn(
                        observation=current_observation,
                        history=history,
                        first_step=(step_idx == 0 and not history.trajectory),
                        task_description=task_descriptions[slot_idx],
                        task_type=task_types[slot_idx],
                        episode_id=episode_ids[slot_idx],
                        active_strategic_node_id=active_strategic_node_id,
                    )
                    action = action.strip() if isinstance(action, str) else ""
                    actions[slot_idx] = action or "look"
                    history.record_action(actions[slot_idx])
                    slot_contexts[slot_idx] = {
                        "history_messages": copy.deepcopy(history_messages),
                        "current_observation": current_observation,
                        "active_strategic_node_id": active_strategic_node_id,
                        "retrieval_state": copy.deepcopy(
                            retrieval_result.to_dict() if retrieval_result is not None else {}
                        ),
                    }

                step_result = self.env_adapter.step(actions)
                next_observations = list(step_result.observations)
                rewards = list(step_result.rewards)
                dones = list(step_result.dones)
                step_infos = [
                    info if isinstance(info, dict) else {}
                    for info in list(step_result.infos or [])
                ]
                if len(step_infos) < batch_size:
                    step_infos.extend({} for _ in range(batch_size - len(step_infos)))

                self.current_step += 1
                if hasattr(self.memory_service, "graph") and self.memory_service.graph is not None:
                    self.memory_service.graph.current_step = self.current_step

                for slot_idx in active_slots:
                    reward = float(rewards[slot_idx]) if slot_idx < len(rewards) else 0.0
                    done = bool(dones[slot_idx]) if slot_idx < len(dones) else False
                    next_obs = str(next_observations[slot_idx] or "") if slot_idx < len(next_observations) else ""
                    self.episode_rewards[slot_idx] += reward
                    reward_histories[slot_idx].append(reward)
                    step_counts[slot_idx] += 1
                    done_flags[slot_idx] = done
                    success_flags[slot_idx] = done and reward > 0
                    observations[slot_idx] = next_obs
                    merged_info = dict(episode_infos[slot_idx]) if slot_idx < len(episode_infos) else {}
                    if slot_idx < len(step_infos):
                        merged_info.update(step_infos[slot_idx])
                    if active_strategic_node_ids[slot_idx] is not None:
                        merged_info["active_strategic_node_id"] = active_strategic_node_ids[slot_idx]
                    if strategic_selection_summaries[slot_idx]:
                        merged_info["active_strategic_node_summary"] = strategic_selection_summaries[
                            slot_idx
                        ]
                    episode_infos[slot_idx] = merged_info
                    self.episode_histories[slot_idx].add_step(next_obs)

                    # The env info does not carry the retrieved memory id, so the
                    # per-step Q-update would otherwise never know which tactical
                    # node the agent actually used. Resolve the active retrieved
                    # tactical node from this slot's retrieval state and feed it
                    # into the Q-update (and downstream formation pipeline).
                    active_memory_id = self._resolve_active_tactical_id(
                        slot_contexts[slot_idx]
                    )
                    q_update_info = dict(step_infos[slot_idx] if slot_idx < len(step_infos) else {})
                    if active_memory_id is not None and "memory_id" not in q_update_info:
                        q_update_info["memory_id"] = active_memory_id

                    source_memory_id, td_error = self._update_step_q_values(
                        task_type=task_types[slot_idx],
                        reward=reward,
                        info=q_update_info,
                        done=done,
                        active_strategic_node_id=active_strategic_node_ids[slot_idx],
                    )
                    slot_context = slot_contexts[slot_idx]
                    if self._should_queue_tactical_candidate(
                        observation=slot_context.get("current_observation", ""),
                        reward=reward,
                        action=actions[slot_idx],
                        td_error=td_error,
                        theta_delta=getattr(self.memory_config, "theta_delta", None),
                    ):
                        retrieval_state = slot_context.get("retrieval_state", {})
                        retrieval_context = "No archived memories."
                        retrieved_ids: List[str] = []
                        if isinstance(retrieval_state, dict):
                            retrieval_context = str(
                                retrieval_state.get("context_text")
                                or retrieval_state.get("retrieved_memories")
                                or "No archived memories."
                            )
                            selected_ids = retrieval_state.get("selected_ids", [])
                            if isinstance(selected_ids, list):
                                retrieved_ids = [
                                    str(item)
                                    for item in selected_ids
                                    if str(item).strip()
                                ]

                        self.pending_formations.append(
                            {
                                "candidate_id": uuid4().hex,
                                "task_type": task_types[slot_idx],
                                "task_description": task_descriptions[slot_idx],
                                "episode_id": episode_ids[slot_idx],
                                "episode_index": slot_idx,
                                "step_index": step_counts[slot_idx],
                                "observation": str(slot_context.get("current_observation", "")),
                                "action": actions[slot_idx],
                                "reward": reward,
                                "td_error": td_error,
                                "history": self._history_messages_to_text(
                                    slot_context.get("history_messages", [])
                                ),
                                "retrieved_memories": retrieval_context,
                                "source_memory_id": source_memory_id,
                                "active_strategic_node_id": slot_context.get(
                                    "active_strategic_node_id"
                                ),
                                "retrieved_ids": retrieved_ids,
                            }
                        )

                    self.results_log.append(
                        {
                            "run_id": self.run_id,
                            "episode_index": slot_idx,
                            "step": step_counts[slot_idx],
                            "task_type": task_types[slot_idx],
                            "task_description": task_descriptions[slot_idx],
                            "episode_id": episode_ids[slot_idx],
                            "action": actions[slot_idx],
                            "observation": next_obs,
                            "reward": reward,
                            "done": done,
                            "active_strategic_node_id": active_strategic_node_id,
                            "info": step_infos[slot_idx] if slot_idx < len(step_infos) else {},
                        }
                    )

            self._update_episode_q_omega(
                task_types=task_types,
                reward_histories=reward_histories,
                step_counts=step_counts,
                done_flags=done_flags,
                step_infos=episode_infos,
                active_strategic_node_ids=active_strategic_node_ids,
            )

            formation_summary = self._commit_pending_formations()
            pruning_summary = self._prune_tactical_nodes()

            if self.sleep_checkpoint is not None and self.mode == "train":
                sleep_summary = self.sleep_checkpoint.check_and_trigger()
            else:
                sleep_summary = None

            episode_summaries = []
            for slot_idx in range(batch_size):
                episode_summaries.append(
                    {
                        "episode_index": slot_idx,
                        "episode_id": episode_ids[slot_idx],
                        "task_type": task_types[slot_idx],
                        "task_description": task_descriptions[slot_idx],
                        "steps": step_counts[slot_idx],
                        "reward": self.episode_rewards[slot_idx],
                        "success": success_flags[slot_idx],
                        "done": done_flags[slot_idx],
                        "active_strategic_node_id": active_strategic_node_ids[slot_idx],
                        "active_strategic_node_summary": strategic_selection_summaries[slot_idx],
                    }
                )

            summary = {
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "mode": self.mode,
                "batch_size": batch_size,
                "max_steps": self.max_steps,
                "episodes": episode_summaries,
                "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
                "mean_steps": float(np.mean(step_counts)) if step_counts else 0.0,
                "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
                "formation": formation_summary,
                "pruning": pruning_summary,
                "sleep_consolidation": sleep_summary,
                "sleep_bootstrap_tactical_min": self.sleep_bootstrap_tactical_min,
            }

            self._report_metrics(
                {
                    "episode/mean_reward": summary["mean_reward"],
                    "episode/mean_steps": summary["mean_steps"],
                    "episode/success_rate": summary["success_rate"],
                    "episode/completed": int(sum(done_flags)),
                    "episode/formation_candidates": formation_summary.get("candidates", 0),
                    "episode/formation_approved": formation_summary.get("approved", 0),
                    "episode/tactical_pruned": pruning_summary.get("pruned", 0),
                }
            )

            summary_path = self.local_cache_dir / "episode_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

            return summary
        finally:
            try:
                self.env_adapter.close()
            except Exception:
                logger.exception("Failed to close episode environment adapter")
            finally:
                self._close_tensorboard_writer()

    def _act_with_retry(
        self,
        *,
        observation: str,
        history_messages: List[Dict[str, Any]],
        first_step: bool,
        active_strategic_node_id: Optional[str],
        current_step: int,
    ) -> AgentDecision:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                decision = self.agent.act(
                    observation=observation,
                    history_messages=history_messages,
                    first_step=first_step,
                    active_strategic_node_id=active_strategic_node_id,
                    current_step=current_step,
                )
                if isinstance(decision, (EnvActionDecision, SkillInvocationDecision)):
                    return decision
                if isinstance(decision, str):
                    text = decision.strip()
                    return EnvActionDecision(action=text or "look", raw_response=decision)
                return EnvActionDecision(action=str(decision), raw_response=str(decision))
            except Exception as exc:
                if attempt >= MAX_RETRIES:
                    logger.error("Agent action failed after %s attempts: %s", attempt, exc)
                    return EnvActionDecision(action="look", raw_response="")
                logger.warning(
                    "Agent action attempt %s/%s failed: %s",
                    attempt,
                    MAX_RETRIES,
                    exc,
                )
                time.sleep(RETRY_DELAY)

        return EnvActionDecision(action="look", raw_response="")

    def _resolve_agent_turn(
        self,
        *,
        observation: str,
        history: EpisodeHistory,
        first_step: bool,
        task_description: str,
        task_type: str,
        episode_id: str,
        active_strategic_node_id: Optional[str],
    ) -> tuple[str, Optional[MemoryRetrievalResult]]:
        latest_retrieval_result: Optional[MemoryRetrievalResult] = None

        for _ in range(MAX_SKILL_INVOCATIONS + 1):
            history_messages = self._history_to_messages(history)
            decision = self._act_with_retry(
                observation=observation,
                history_messages=history_messages,
                first_step=first_step,
                active_strategic_node_id=active_strategic_node_id,
                current_step=self.current_step,
            )
            history.append_message(decision.as_message())

            if isinstance(decision, EnvActionDecision):
                action = decision.action.strip() or "look"
                return action, latest_retrieval_result

            if isinstance(decision, SkillInvocationDecision):
                if decision.skill_name.strip() != "memory_retrieval":
                    history.append_message(
                        {
                            "role": "tool",
                            "name": decision.skill_name or "skill",
                            "content": (
                                f"Unsupported skill: {decision.skill_name or 'unknown'}. "
                                "Available skill: memory_retrieval."
                            ),
                        }
                    )
                    continue

                query_override = None
                for key in ("query", "query_text", "text"):
                    value = decision.arguments.get(key)
                    if isinstance(value, str) and value.strip():
                        query_override = value.strip()
                        break

                try:
                    retrieval_result = self.memory_retrieval_skill.retrieve(
                        task_description=task_description,
                        observation=observation,
                        history_messages=history_messages,
                        task_type=task_type,
                        episode_id=episode_id,
                        active_strategic_node_id=active_strategic_node_id,
                        current_step=self.current_step,
                        query_override=query_override,
                    )
                    latest_retrieval_result = retrieval_result
                    history.append_message(
                        retrieval_result.to_tool_message(skill_name=decision.skill_name)
                    )
                    if hasattr(self.agent, "record_memory_retrieval"):
                        try:
                            self.agent.record_memory_retrieval(retrieval_result)
                        except Exception:
                            logger.debug(
                                "Agent failed to record memory retrieval result",
                                exc_info=True,
                            )
                except Exception as exc:
                    logger.warning("Memory retrieval failed: %s", exc)
                    history.append_message(
                        {
                            "role": "tool",
                            "name": decision.skill_name or "memory_retrieval",
                            "content": f"Memory retrieval failed: {exc}",
                        }
                    )
                    latest_retrieval_result = None
                continue

            history.append_message(
                {
                    "role": "tool",
                    "name": "agent",
                    "content": "Unsupported agent decision; defaulting to environment action.",
                }
            )

        logger.warning(
            "Agent did not produce an environment action after %s skill turns; defaulting to look",
            MAX_SKILL_INVOCATIONS,
        )
        return "look", latest_retrieval_result

    def _build_retrieval_query(
        self,
        *,
        task_description: str,
        observation: str,
        history_messages: List[Dict[str, str]],
    ) -> str:
        history_text = self._history_messages_to_text(history_messages)
        parts = [
            f"Task: {task_description.strip()}",
            f"Observation: {observation.strip()}",
        ]
        if history_text:
            parts.append(f"History: {history_text}")
        return "\n".join(part for part in parts if part.strip())

    def _history_messages_to_text(self, history_messages: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for message in history_messages[-10:]:
            role = str(message.get("role", "user")).strip()
            content = str(message.get("content", "")).strip()
            if content:
                name = str(message.get("name", "")).strip()
                if role == "tool" and name:
                    lines.append(f"{role}[{name}]: {content}")
                else:
                    lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _retrieve_memory_context(
        self,
        *,
        query: str,
        task_type: str,
        active_strategic_node_id: Optional[str],
    ) -> str:
        try:
            tau = float(getattr(self.rl_config, "sim_threshold", getattr(self.rl_config, "tau", 0.0)))
        except Exception:
            tau = 0.0

        try:
            result, _ = self.memory_service.retrieve_query(
                query,
                k=self.retrieve_k,
                threshold=tau,
                task_type_dominant=task_type,
                active_strategic_node_id=active_strategic_node_id,
            )
        except Exception as exc:
            logger.warning("Memory retrieval failed: %s", exc)
            return "No archived memories."

        selected = (result or {}).get("selected", [])
        if not isinstance(selected, list) or not selected:
            return "No archived memories."

        parts: List[str] = []
        for idx, item in enumerate(selected, 1):
            if not isinstance(item, dict):
                continue
            mem_id = str(item.get("memory_id") or item.get("id") or f"memory-{idx}")
            content = str(item.get("content") or "").strip()
            score = float(item.get("score", 0.0) or 0.0)
            if not content:
                continue
            parts.append(f"{idx}. [{mem_id}] (score={score:.3f}) {content}")

        return "\n".join(parts) if parts else "No archived memories."

    def _history_to_messages(self, history: EpisodeHistory) -> List[Dict[str, str]]:
        messages = history.get_messages()
        if messages:
            return messages

        fallback_messages: List[Dict[str, str]] = []
        for step in history.trajectory:
            action = str(step.get("action", "")).strip()
            observation = str(step.get("observation", "")).strip()
            content = "\n".join(
                part for part in [f"Action: {action}" if action else "", f"Observation: {observation}" if observation else ""]
                if part
            )
            if content:
                fallback_messages.append({"role": "user", "content": content})
        return fallback_messages

    def _infer_task_description(self, observation: str, info: Dict[str, Any]) -> str:
        for key in (
            "task_description",
            "question",
            "prompt",
            "instruction",
            "goal",
            "description",
            "text",
        ):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        obs = str(observation).strip()
        return obs

    def _infer_task_type(self, index: int, info: Dict[str, Any]) -> str:
        for key in ("task_type", "category", "benchmark", "domain"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        mode = str(getattr(self.memory_config, "task_type_mode", "explicit")).lower()
        if mode == "benchmark":
            mapped = self._benchmark_task_type()
            if mapped is not None:
                return mapped
        return f"episode_{index}"

    def _benchmark_task_type(self) -> Optional[str]:
        """Coarse benchmark-level taxonomy (W5) when no explicit task type.

        alfworld -> embodied, bcb -> coding, hle -> reasoning, llb -> lifelong.
        Returns None when the benchmark cannot be inferred from the experiment name.
        """
        name = (self.experiment_name or "").lower()
        if "alf" in name:
            return "embodied"
        if "bcb" in name or "bigcode" in name:
            return "coding"
        if "hle" in name:
            return "reasoning"
        if "llb" in name or "lifelong" in name:
            return "lifelong"
        return None

    def _infer_episode_id(self, index: int, info: Dict[str, Any]) -> str:
        for key in ("episode_id", "id", "sample_id", "task_id", "gamefile"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"{self.experiment_name}_{self.run_id}_{index}"

    def _report_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = dict(metrics)
        self.metrics_history.append(payload)
        self._report_tensorboard(payload)

        try:
            from ray.air import session  # type: ignore

            if session.get_session() is not None:
                session.report(payload)
                return
        except Exception:
            pass

        try:
            from ray import train as ray_train  # type: ignore

            if hasattr(ray_train, "report"):
                ray_train.report(payload)
                return
        except Exception:
            pass

        logger.info("%s metrics: %s", self.metrics_namespace, payload)

    def _init_tensorboard_writer(self, tensorboard_log_dir: Optional[str]) -> Any:
        if not tensorboard_log_dir:
            return None
        tb_path = Path(tensorboard_log_dir)
        tb_path.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is None:
            logger.info(
                "TensorBoard is not available; skipping writer for %s",
                tb_path,
            )
            return None
        writer = SummaryWriter(log_dir=str(tb_path))
        logger.info("TensorBoard logs will be saved to: %s", tb_path)
        return writer

    def _report_tensorboard(self, metrics: Dict[str, Any]) -> None:
        writer = getattr(self, "tensorboard_writer", None)
        if writer is None:
            return
        step = int(self.current_step)
        for key, value in metrics.items():
            if isinstance(value, bool):
                writer.add_scalar(key, int(value), step)
            elif isinstance(value, (int, float)):
                writer.add_scalar(key, float(value), step)

    def _close_tensorboard_writer(self) -> None:
        writer = getattr(self, "tensorboard_writer", None)
        if writer is None:
            return
        try:
            writer.flush()
        except Exception:
            logger.debug("TensorBoard writer flush failed", exc_info=True)
        try:
            writer.close()
        except Exception:
            logger.debug("TensorBoard writer close failed", exc_info=True)

    def _update_step_q_values(
        self,
        *,
        task_type: str,
        reward: float,
        info: Dict[str, Any],
        done: bool,
        active_strategic_node_id: Optional[str],
    ) -> tuple[Optional[str], Optional[float]]:
        node_id = self._resolve_tactical_node_id(info)
        if node_id is None:
            return None, None

        node = self.memory_service.graph.nodes.get(node_id)
        if node is None or not getattr(node, "is_tactical", False):
            return None, None

        current_value = float((node.Q or {}).get(task_type, 0.0))
        next_value = 0.0 if done else self._estimate_next_tactical_value(
            task_type=task_type,
            active_strategic_node_id=active_strategic_node_id,
        )
        gamma = float(getattr(self.memory_config, "gamma", 0.95))
        alpha = float(getattr(self.memory_config, "alpha", 0.1))
        td_error = compute_td_error(
            reward,
            gamma=gamma,
            next_value=next_value,
            current_value=current_value,
        )
        node.Q[task_type] = apply_q_update(
            current_value,
            td_error,
            alpha=alpha,
        )
        node.n[task_type] = int(node.n.get(task_type, 0) or 0) + 1
        node.last_accessed_step = self.current_step
        if hasattr(self.memory_service.graph, "refresh_decay_rate"):
            self.memory_service.graph.refresh_decay_rate(node)
        else:
            node.recompute_decay_rate(
                lambda_base=float(getattr(self.memory_config, "lambda_base", 0.0) or 0.0),
                epsilon=float(getattr(self.memory_config, "epsilon_decay", 0.01)),
                lambda_shrink=float(getattr(self.memory_config, "lambda_shrink", 10.0)),
            )

        self._report_metrics(
            {
                "episode/td_error": td_error,
                "episode/tactical_q": float(node.Q.get(task_type, 0.0)),
                "episode/tactical_salience": float(get_q_salience(node, lambda_shrink=float(getattr(self.memory_config, "lambda_shrink", 10.0)))),
            }
        )
        self.memory_service.persist_node_state(node)

        return node_id, td_error

    def _estimate_next_tactical_value(
        self,
        *,
        task_type: str,
        active_strategic_node_id: Optional[str],
    ) -> float:
        graph = getattr(self.memory_service, "graph", None)
        if graph is None:
            return 0.0

        if active_strategic_node_id is not None:
            candidate_ids = self._graph_child_ids(graph, active_strategic_node_id)
            candidate_nodes = [
                graph.nodes[node_id]
                for node_id in candidate_ids
                if node_id in graph.nodes
                and getattr(graph.nodes[node_id], "is_tactical", False)
            ]
        else:
            candidate_nodes = [
                node
                for node in graph.nodes.values()
                if getattr(node, "is_tactical", False)
            ]

        if not candidate_nodes:
            return 0.0

        return max(
            float((node.Q or {}).get(task_type, 0.0) or 0.0)
            for node in candidate_nodes
        )

    @staticmethod
    def _graph_child_ids(graph: Any, parent_id: str) -> set[str]:
        if hasattr(graph, "child_ids"):
            try:
                return set(graph.child_ids(parent_id))
            except Exception:
                logger.debug("Failed to query graph child ids", exc_info=True)

        return {
            getattr(node, "id", "")
            for node in getattr(graph, "nodes", {}).values()
            if getattr(node, "parent_id", None) == parent_id
        }

    def _update_episode_q_omega(
        self,
        *,
        task_types: List[str],
        reward_histories: List[List[float]],
        step_counts: List[int],
        done_flags: List[bool],
        step_infos: List[Dict[str, Any]],
        active_strategic_node_ids: List[Optional[str]],
    ) -> None:
        gamma_omega = float(getattr(self.memory_config, "gamma_omega", 0.95))
        alpha_omega = float(getattr(self.memory_config, "alpha_omega", 0.1))
        # W4 single-discount ablation: when strategic_discount_mode == "shared",
        # collapse the strategic discount onto the tactical gamma so the
        # separate-gamma claim can be tested against the single-gamma control.
        if str(getattr(self.memory_config, "strategic_discount_mode", "separate")).lower() == "shared":
            gamma_omega = float(getattr(self.memory_config, "gamma", gamma_omega))
        for slot_idx, rewards in enumerate(reward_histories):
            node_id = active_strategic_node_ids[slot_idx] if slot_idx < len(active_strategic_node_ids) else None
            if node_id is None:
                node_id = self._resolve_strategic_node_id(
                    step_infos[slot_idx] if slot_idx < len(step_infos) else {}
                )
            if node_id is None:
                continue

            node = self.memory_service.graph.nodes.get(node_id)
            if node is None or not getattr(node, "is_strategic", False):
                continue

            episode_return = 0.0
            for t, reward in enumerate(rewards):
                episode_return += (gamma_omega ** t) * float(reward)

            for task_type in {task_types[slot_idx]}:
                current_value = float((node.Q_omega or {}).get(task_type, 0.0))
                td_error = compute_td_error(
                    episode_return,
                    gamma=1.0,
                    next_value=0.0,
                    current_value=current_value,
                )
                node.Q_omega[task_type] = apply_q_update(
                    current_value,
                    td_error,
                    alpha=alpha_omega,
                )
                node.n_omega[task_type] = int(node.n_omega.get(task_type, 0) or 0) + 1

            self.memory_service.persist_node_state(node)

            # Feed empirical episode-length statistics for finite-horizon
            # Q^Omega init of future spawned scaffolds (spec §3.5, W3).
            if hasattr(self.memory_service, "record_episode_length"):
                try:
                    self.memory_service.record_episode_length(
                        task_types[slot_idx],
                        int(step_counts[slot_idx] if slot_idx < len(step_counts) else len(rewards)),
                    )
                except Exception:
                    logger.debug("Failed to record episode length", exc_info=True)

            self._report_metrics(
                {
                    "episode/omega_return": episode_return,
                    "episode/omega_q": get_q_omega_salience(
                        node,
                        lambda_shrink=float(getattr(self.memory_config, "lambda_shrink", 10.0)),
                    ),
                }
            )

    def _select_strategic_scaffold(
        self,
        *,
        task_description: str,
        task_type: str,
        observation: str,
        history_messages: List[Dict[str, Any]],
        episode_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        if not self._has_strategic_scaffolds():
            return None, None

        try:
            result, _ = self.memory_service.retrieve_query(
                task_description,
                k=self.strategic_k,
                depth=1,
                task_type_dominant=task_type,
            )
        except Exception as exc:
            logger.warning("Strategic retrieval failed for episode=%s: %s", episode_id, exc)
            return None, None

        candidates = (result or {}).get("selected", [])
        if not isinstance(candidates, list) or not candidates:
            return None, None

        llm = getattr(self.agent, "llm", None) or self.llm_provider
        if llm is None:
            candidate = candidates[0] if candidates else {}
            chosen_id = self._coerce_optional_strategy_id(candidate)
            return chosen_id, self._strategy_summary(candidate)

        prompt_messages = [
            {"role": "system", "content": agent_prompts.STRATEGIC_SELECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": agent_prompts.STRATEGIC_SELECTION_USER_PROMPT.format(
                    task_description=task_description,
                    task_type=task_type,
                    observation=observation,
                    history=self._history_messages_to_text(history_messages)
                    if history_messages
                    else "You are at the beginning of the task. No steps taken yet.",
                    strategies=self._render_strategic_candidates(candidates),
                ),
            },
        ]

        try:
            response = llm.generate(prompt_messages, temperature=0.0)
        except Exception as exc:
            logger.warning("Strategic selection LLM failed for episode=%s: %s", episode_id, exc)
            response = ""

        chosen_id = self._parse_strategic_selection_response(response, candidates)
        if chosen_id is None:
            candidate = candidates[0] if candidates else {}
            chosen_id = self._coerce_optional_strategy_id(candidate)

        summary = None
        for candidate in candidates:
            if self._coerce_optional_strategy_id(candidate) == chosen_id:
                summary = self._strategy_summary(candidate)
                break

        return chosen_id, summary

    @staticmethod
    def _render_strategic_candidates(candidates: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, candidate in enumerate(candidates, 1):
            node_id = str(candidate.get("memory_id") or candidate.get("id") or "").strip()
            summary = str(candidate.get("content") or "").strip() or "No summary available."
            score = float(candidate.get("score", 0.0) or 0.0)
            lines.append(f"{idx}. id={node_id} score={score:.3f} summary={summary}")
        return "\n".join(lines) if lines else "No strategic candidates."

    @staticmethod
    def _parse_strategic_selection_response(
        response: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[str]:
        text = (response or "").strip()
        if not text:
            return None

        payload: Optional[Dict[str, Any]] = None
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    loaded = json.loads(text[start : end + 1])
                    if isinstance(loaded, dict):
                        payload = loaded
                except Exception:
                    payload = None

        if payload is None:
            return None

        for key in ("strategy_id", "selected_id", "id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                candidate_ids = {
                    str(candidate.get("memory_id") or candidate.get("id") or "").strip()
                    for candidate in candidates
                }
                chosen = value.strip()
                if chosen in candidate_ids:
                    return chosen
        return None

    @staticmethod
    def _coerce_optional_strategy_id(candidate: Dict[str, Any]) -> Optional[str]:
        node_id = str(candidate.get("memory_id") or candidate.get("id") or "").strip()
        return node_id or None

    @staticmethod
    def _strategy_summary(candidate: Dict[str, Any]) -> Optional[str]:
        summary = str(candidate.get("content") or "").strip()
        return summary or None

    def _resolve_strategic_node_id(self, info: Dict[str, Any]) -> Optional[str]:
        for key in ("active_strategic_node_id", "strategic_node_id", "omega_id"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        strategic_nodes = self.memory_service.graph.nodes_at_depth(1)
        if len(strategic_nodes) == 1:
            return strategic_nodes[0].id
        return None

    def _resolve_tactical_node_id(self, info: Dict[str, Any]) -> Optional[str]:
        for key in ("memory_id", "active_memory_id", "tactical_node_id", "selected_node_id"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_active_tactical_id(self, slot_context: Dict[str, Any]) -> Optional[str]:
        """Return the tactical node id the agent retrieved for this slot, if any.

        The env adapter does not know which memory the agent used, so without
        this the per-step Q-update and TD-driven formation pipeline never fire
        in the real loop. We look up the retrieval state captured during the
        agent turn and return the first selected id that is currently a
        *tactical* node in the graph.
        """
        retrieval_state = slot_context.get("retrieval_state") if isinstance(slot_context, dict) else None
        if not isinstance(retrieval_state, dict):
            return None
        raw_ids = retrieval_state.get("selected_ids")
        if not isinstance(raw_ids, list):
            return None
        graph = getattr(self.memory_service, "graph", None)
        for raw in raw_ids:
            node_id = str(raw or "").strip()
            if not node_id:
                continue
            if graph is not None:
                node = graph.nodes.get(node_id)
                if node is None or not getattr(node, "is_tactical", False):
                    continue
            return node_id
        return None

    def _has_strategic_scaffolds(self) -> bool:
        graph = getattr(self.memory_service, "graph", None)
        if graph is None:
            return False
        try:
            return bool(graph.nodes_at_depth(1))
        except Exception:
            logger.debug("Failed to inspect strategic scaffold availability", exc_info=True)
        return False

    def _should_queue_tactical_candidate(
        self,
        *,
        observation: Any,
        reward: float,
        action: Any,
        td_error: Optional[float],
        theta_delta: Optional[float],
    ) -> bool:
        if td_error is None:
            return False
        threshold = float(theta_delta) if theta_delta is not None else 0.0
        if td_error <= threshold:
            return False
        if float(reward) <= 0.0:
            return False
        if not str(observation or "").strip():
            return False
        if not str(action or "").strip():
            return False
        return True

    def _commit_pending_formations(self) -> Dict[str, Any]:
        candidates_raw = list(self.pending_formations)
        self.pending_formations = []
        if not candidates_raw:
            return {"candidates": 0, "approved": 0, "created_nodes": [], "skipped": False}

        if self.formation_judge is None:
            logger.warning(
                "Formation judge is unavailable; skipping %s pending tactical candidates",
                len(candidates_raw),
            )
            return {
                "candidates": len(candidates_raw),
                "approved": 0,
                "created_nodes": [],
                "skipped": True,
            }

        candidates = [
            TacticalFormationCandidate(**candidate)
            for candidate in candidates_raw
            if self._should_queue_tactical_candidate(
                observation=candidate.get("observation", ""),
                reward=float(candidate.get("reward", 0.0) or 0.0),
                action=candidate.get("action", ""),
                td_error=float(candidate.get("td_error", 0.0) or 0.0),
                theta_delta=getattr(self.memory_config, "theta_delta", None),
            )
        ]
        if not candidates:
            return {
                "candidates": len(candidates_raw),
                "approved": 0,
                "created_nodes": [],
                "skipped": True,
                "filtered": True,
            }
        try:
            decisions = self.formation_judge.judge_candidates(candidates)
        except Exception as exc:
            logger.warning("Formation judge failed; skipping tactical storage: %s", exc)
            return {
                "candidates": len(candidates_raw),
                "approved": 0,
                "created_nodes": [],
                "skipped": True,
                "error": str(exc),
            }

        decisions_by_id = {decision.candidate_id: decision for decision in decisions}
        created_nodes: List[str] = []
        for candidate in candidates:
            decision = decisions_by_id.get(candidate.candidate_id)
            if decision is None or not decision.approved:
                continue

            parent_id = candidate.active_strategic_node_id or self.memory_service.graph.root_id
            node_id = uuid4().hex
            evidence_ids = [candidate.candidate_id]
            if candidate.source_memory_id:
                evidence_ids.append(candidate.source_memory_id)

            summary_content = decision.summary or candidate.fallback_summary()
            summary_writer = self.tactical_summary_writer
            if summary_writer is not None:
                try:
                    summary_draft = summary_writer.summarize_candidate(candidate)
                    summary_content = summary_writer.format_summary(summary_draft) or summary_content
                except Exception as exc:
                    logger.warning(
                        "Tactical summary generation failed; using judge summary instead: %s",
                        exc,
                    )

            self.memory_service.add_node_from_text(
                id=node_id,
                content=summary_content,
                task_type_dominant=candidate.task_type,
                t_create=int(self.current_step),
                depth=2,
                parent_id=parent_id,
                evidence_ids=evidence_ids,
                last_accessed_step=int(self.current_step),
            )
            created_nodes.append(node_id)

        return {
            "candidates": len(candidates_raw),
            "approved": len(created_nodes),
            "created_nodes": created_nodes,
            "skipped": False,
        }

    def _prune_tactical_nodes(self) -> Dict[str, Any]:
        theta_prune = getattr(self.memory_config, "theta_prune", None)
        if theta_prune is None:
            return {"pruned": 0, "pruned_node_ids": [], "theta_prune": None}

        pruned_node_ids = self.memory_service.prune_tactical_nodes(
            current_step=self.current_step,
            theta_prune=float(theta_prune),
        )
        return {
            "pruned": len(pruned_node_ids),
            "pruned_node_ids": pruned_node_ids,
            "theta_prune": float(theta_prune),
        }
