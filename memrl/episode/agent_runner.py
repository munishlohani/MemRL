from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from memrl.memory.episodic_bank import EpisodicRecord
from memrl.service.memory_service import MemoryService
from typing import Any, Dict, List, Optional, Sequence
from memrl.service.sleep_consolidation.checkpoint import SleepConsolidationCheckpoint
from memrl.service.formation_judger import (
    TacticalFormationCandidate,
    TacticalSummaryWriter,
)
from memrl.skills.memory_retrieval import MemoryRetrievalResult, MemoryRetrievalSkill
from memrl.providers.base import BaseLLM
from memrl.utils.q_utils import (
    apply_q_update,
    compute_advantage,
    compute_mc_return_to_go,
    get_q_omega_salience,
    get_q_salience,
)
from .env_adapter import EpisodeEnvAdapter
from memrl.utils.event_logging import log_event

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
        run_dir: Optional[Path] = None,
        retrieve_k: int = 1,
        batch_size: int = 1,
        max_steps: int = 1,
        llm_provider: Optional[BaseLLM] = None,
        strategic_k: int = 3,
        max_skill_invocations: int = MAX_SKILL_INVOCATIONS,
        skill_budget_per_episode: Optional[int] = None,
        tensorboard_log_dir: Optional[str] = None,
        skill_contract_path: Optional[str] = None,
        auto_inject_memory: bool = False,
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
        self.tactical_summary_writer = (
            TacticalSummaryWriter(llm_provider) if llm_provider is not None else None
        )
        self.memory_retrieval_skill = MemoryRetrievalSkill(
            memory_service=memory_service,
            llm_provider=llm_provider,
            retrieve_k=retrieve_k,
            rl_config=self.rl_config,
            contract_path=skill_contract_path,
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
        # Per-run cap on skill (memory_retrieval) invocations within a single
        # _resolve_agent_turn call. Defaults to the module constant (3,
        # ALFWorld's existing budget); benchmarks with a much smaller
        # per-episode action budget (BCB: max_steps=1) pass a lower cap so
        # the agent can't spend its one real turn on repeated retrieval
        # attempts instead of ever submitting an answer.
        self.max_skill_invocations = max(0, int(max_skill_invocations))
        # Per-EPISODE budget (all steps combined), distinct from
        # max_skill_invocations above (which only bounds retries within a
        # single _resolve_agent_turn call, i.e. per-step). None = unlimited.
        # Static: starts at skill_budget_per_episode, decrements by 1 on
        # each actual retrieval, and once it hits 0 stays there for the
        # rest of the episode -- no regeneration. The agent is told its
        # current remaining count every turn (see CustomAgent._build_messages)
        # so it can pace itself instead of spending it all in the first few
        # steps; that visibility is what's dynamic, not the budget itself.
        self.skill_budget_per_episode = (
            int(skill_budget_per_episode) if skill_budget_per_episode is not None else None
        )
        self._skill_budget_remaining: List[int] = []
        # Per-slot retrieval bookkeeping, surfaced on the episode record.
        # Without it the only evidence a retrieval happened is a pair of INFO
        # lines from MemoryRetrievalSkill, and a budget-refused invocation
        # emits nothing at all -- making "the agent never retrieved" and "the
        # agent asked but was refused" indistinguishable after the fact.
        self._skill_retrievals: List[int] = []
        self._skill_refused_budget: List[int] = []
        # Retrieve once up front instead of waiting for the agent to ask.
        # Intended for single-step benchmarks where the agent has no
        # information on which to base that choice -- see
        # _auto_inject_memory. Off by default: multi-step benchmarks keep
        # retrieval agent-driven.
        self.auto_inject_memory = bool(auto_inject_memory)

        # run_dir defaults to a generic "episode" subtree (benchmark-neutral,
        # since this runner is shared across ALFWorld/BabyAI/HLE/etc.), but a
        # caller that already computed its own benchmark-specific run
        # directory (e.g. run_alfworld.py's results/alfworld/exp_.../) can
        # pass it directly so episodes.jsonl/metrics.jsonl/etc. land
        # alongside that run's skill DB, tensorboard events, and token logs
        # instead of in a separate, run_id-mismatched tree.
        self.run_dir = Path(run_dir) if run_dir is not None else (
            self.output_dir / "episode" / f"exp_{self.experiment_name}_{self.run_id}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.local_cache_dir = self.run_dir / "local_cache"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_writer = self._init_tensorboard_writer(tensorboard_log_dir)

        self.current_step = 0
        self._episode_counter = 0
        self.results_log: List[Dict[str, Any]] = []
        self.episode_histories = [EpisodeHistory() for _ in range(self.batch_size)]
        self.pending_formations: List[Dict[str, Any]] = []
        self.episode_rewards: List[float] = []
        self.active_strategic_node_ids: List[Optional[str]] = []
        self.active_strategic_node_summaries: List[Optional[str]] = []
        self.sleep_bootstrap_tactical_min = getattr(self.memory_config, "n_sleep", None)
        self.metrics_namespace = f"episode/{self.experiment_name}"
        self.metrics_history: List[Dict[str, Any]] = []

        # Cumulative (running, across the whole training run) metrics state.
        # Per-batch metrics alone hide slow trends (baseline convergence,
        # differential pruning, task-type collapse) that only show up over
        # many episodes -- these dicts/counters accumulate across run() calls.
        self._task_type_success_counts: Dict[str, int] = {}
        self._task_type_total_counts: Dict[str, int] = {}
        self._task_type_length_success: Dict[str, List[int]] = {}
        self._task_type_length_failure: Dict[str, List[int]] = {}
        self._strategic_selection_counts: Dict[str, int] = {}
        self._cumulative_nodes_created = 0
        self._episodes_completed_cumulative = 0
        self._cumulative_pruned_count = 0
        self._cumulative_pruned_by_task_type: Dict[str, int] = {}
        # §P2.6.1 telemetry: section index gives the "vertical marker" for
        # section boundaries when overlaid on episode/mean_reward (one run()
        # call == one section/mini-batch in this architecture); cumulative
        # sleep-action counts answer "has absorb ever fired?" from a single
        # run-lifetime scalar instead of requiring correlation across rare,
        # easy-to-miss per-event snapshots.
        self._section_index = 0
        self._cumulative_sleep_action_counts: Dict[str, int] = {}
        self._seed_known_task_types()

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
        # One shallow-copied agent per slot: CustomAgent.reset() mutates
        # instance state (task_description/task_type/episode_id/_trajectory)
        # that act() reads back implicitly, so slots sharing one agent
        # instance would race once turn resolution is thread-pooled below.
        # Shallow copy isolates that per-episode scalar state per slot while
        # still sharing the (thread-safe) llm/memory_retrieval_skill refs.
        agent_slots = [copy.copy(self.agent) for _ in range(batch_size)]
        self.episode_histories = [EpisodeHistory() for _ in range(batch_size)]
        # Pin the room's initial description into history explicitly --
        # env.reset()'s observation (which includes the room's receptacle
        # listing, e.g. "Looking quickly around you, you see...") is
        # otherwise shown only once, as the first turn's "Current
        # Observation", then never persisted anywhere: EpisodeHistory.
        # add_step() only pairs a NEW observation with the PRECEDING action,
        # so the reset observation itself never enters history.messages or
        # .trajectory. Without this, the room layout silently vanishes from
        # the agent's context after turn 1 unless it happens to re-issue
        # "look" (spec: agent flailing on receptacle names it invented).
        for slot_idx in range(batch_size):
            self.episode_histories[slot_idx].append_message(
                {
                    "role": "system",
                    "content": f"Initial Room Description: {observations[slot_idx]}",
                }
            )
        self.pending_formations = []
        self.episode_rewards = [0.0 for _ in range(batch_size)]
        self._skill_budget_remaining = [
            self.skill_budget_per_episode if self.skill_budget_per_episode is not None else 0
            for _ in range(batch_size)
        ]
        self._skill_retrievals = [0 for _ in range(batch_size)]
        self._skill_refused_budget = [0 for _ in range(batch_size)]
        episode_numbers = [self._next_episode_number() for _ in range(batch_size)]
        episode_candidate_buffers: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        reward_histories = [[] for _ in range(batch_size)]
        active_tactical_visits: List[List[Optional[str]]] = [[] for _ in range(batch_size)]
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
                self._strategic_selection_counts[selected_id] = (
                    self._strategic_selection_counts.get(selected_id, 0) + 1
                )
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

                def _process_slot(slot_idx: int) -> None:
                    # LLM calls are I/O-bound (network latency), so slots run
                    # concurrently on a thread pool -- one agent.act() /
                    # memory retrieval per active slot per step, matching the
                    # base-template runner's per-step ThreadPoolExecutor
                    # fan-out (spec P2.9). Each slot only reads/writes its own
                    # index into actions/slot_contexts/active_strategic_node_ids
                    # and its own agent_slots[slot_idx]/episode_histories[slot_idx],
                    # so there is no cross-slot mutation to race on.
                    try:
                        history = self.episode_histories[slot_idx]
                        history_messages = self._history_to_messages(history)
                        current_observation = str(observations[slot_idx] or "")
                        active_strategic_node_id = active_strategic_node_ids[slot_idx]
                        if active_strategic_node_id is None and has_strategic_scaffolds:
                            active_strategic_node_id = self._resolve_strategic_node_id(
                                episode_infos[slot_idx] if slot_idx < len(episode_infos) else {}
                            )
                            active_strategic_node_ids[slot_idx] = active_strategic_node_id
                        agent_slots[slot_idx].reset(
                            task_description=task_descriptions[slot_idx],
                            task_type=task_types[slot_idx],
                            episode_id=episode_ids[slot_idx],
                        )
                        admissible_commands = list(
                            (episode_infos[slot_idx] if slot_idx < len(episode_infos) else {}).get(
                                "admissible_commands"
                            )
                            or []
                        )
                        action, retrieval_result = self._resolve_agent_turn(
                            agent=agent_slots[slot_idx],
                            observation=current_observation,
                            history=history,
                            first_step=(step_idx == 0 and not history.trajectory),
                            task_description=task_descriptions[slot_idx],
                            task_type=task_types[slot_idx],
                            episode_id=episode_ids[slot_idx],
                            active_strategic_node_id=active_strategic_node_id,
                            admissible_commands=admissible_commands,
                            slot_idx=slot_idx,
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
                    except Exception:
                        logger.exception("Slot %s turn resolution failed; defaulting to look", slot_idx)
                        actions[slot_idx] = "look"

                with ThreadPoolExecutor(max_workers=len(active_slots)) as executor:
                    futures = [executor.submit(_process_slot, slot_idx) for slot_idx in active_slots]
                    for future in as_completed(futures):
                        future.result()

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
                    # Tactical Q is no longer updated inline (bootstrap TD is
                    # gone). The retrieved node id is only recorded here so
                    # the end-of-episode MC return-to-go update knows which
                    # tactical node was active at this step (spec §3.2, §3.7).
                    source_memory_id = self._resolve_tactical_node_id(q_update_info)
                    active_tactical_visits[slot_idx].append(source_memory_id)
                    slot_context = slot_contexts[slot_idx]
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

                    episode_candidate_buffers[slot_idx].append(
                        {
                            "candidate_id": uuid4().hex,
                            "task_type": task_types[slot_idx],
                            "task_description": task_descriptions[slot_idx],
                            "episode_id": episode_ids[slot_idx],
                            "episode_index": episode_numbers[slot_idx],
                            "episode_slot_index": slot_idx,
                            "step_index": step_counts[slot_idx],
                            "observation": str(slot_context.get("current_observation", "")),
                            "action": actions[slot_idx],
                            "reward": reward,
                            "advantage": None,
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
                            "episode_index": episode_numbers[slot_idx],
                            "episode_slot_index": slot_idx,
                            "step": step_counts[slot_idx],
                            "global_step": self.current_step,
                            "task_type": task_types[slot_idx],
                            "task_description": task_descriptions[slot_idx],
                            "episode_id": episode_ids[slot_idx],
                            "action": actions[slot_idx],
                            "observation": next_obs,
                            "reward": reward,
                            "done": done,
                            "active_strategic_node_id": active_strategic_node_ids[slot_idx],
                            "info": step_infos[slot_idx] if slot_idx < len(step_infos) else {},
                        }
                    )

            # build_memory is the master switch for the whole "write" side of
            # memory, independent of self.mode -- mode only selects the
            # ALFWorld data split (train / eval_in_distribution /
            # eval_out_of_distribution), it is not a "is this training run"
            # flag. When False, memory is used (retrieval/selection still
            # run every step, unaffected) but never built: no tactical
            # formation, no Q-value/baseline updates, no pruning, no sleep
            # consolidation -- the graph is read-only for the whole episode.
            # This is what an evaluation run against a fixed, already-built
            # skill graph needs (see also memory.reuse_skill_db).
            build_memory = bool(getattr(self.memory_config, "build_memory", True))

            # Stage-1 gate (§4.1) must read the tactical baseline b(t_k)
            # before this episode's own return updates it, so it runs before
            # _update_episode_tactical_q (which performs that update).
            if build_memory:
                formation_gate_stats = self._queue_episode_tactical_candidates(
                    reward_histories=reward_histories,
                    candidate_buffers=episode_candidate_buffers,
                    success_flags=success_flags,
                )
            else:
                formation_gate_stats = self._new_formation_gate_stats()

            # Working-set protocol (§5.3): step-level Q-updates mutate nodes
            # in memory only; touched nodes are collected here and flushed
            # to SQLite once, after pruning, instead of one transaction per
            # node per step.
            dirty_nodes: Dict[str, Any] = {}
            if build_memory:
                self._update_episode_tactical_q(
                    task_types=task_types,
                    reward_histories=reward_histories,
                    active_tactical_visits=active_tactical_visits,
                    dirty_nodes=dirty_nodes,
                )

                self._update_episode_q_omega(
                    task_types=task_types,
                    reward_histories=reward_histories,
                    step_infos=episode_infos,
                    active_strategic_node_ids=active_strategic_node_ids,
                    dirty_nodes=dirty_nodes,
                )

                self._queue_failed_episode_reflections(
                    task_descriptions=task_descriptions,
                    success_flags=success_flags,
                    active_strategic_node_ids=active_strategic_node_ids,
                )

            # Skipping _queue_episode_tactical_candidates above already
            # leaves self.pending_formations empty, so this is naturally a
            # no-op when build_memory is False -- no separate branch needed.
            formation_summary = self._commit_pending_formations()
            if build_memory:
                pruning_summary = self._prune_tactical_nodes()
            else:
                pruning_summary = {
                    "pruned": 0,
                    "pruned_node_ids": [],
                    "theta_prune": None,
                    "pruned_by_task_type": {},
                }
            self._flush_dirty_nodes(dirty_nodes)

            if self.sleep_checkpoint is not None and build_memory:
                sleep_summary = self.sleep_checkpoint.check_and_trigger()
            else:
                sleep_summary = None

            # A slot is "duplicate" when the env_adapter had to wrap around
            # and re-dispatch an already-seen game this epoch to fill a
            # fixed batch_size that doesn't evenly divide the split size
            # (see AlfWorldEpisodeEnvAdapter.reset_epoch_tracking). Excluded
            # here from both the aggregated scalars and episodes.jsonl so a
            # handful of repeated games at the tail of an epoch don't get
            # double-counted in eval metrics.
            duplicate_flags = [
                bool((episode_infos[slot_idx] if slot_idx < len(episode_infos) else {}).get("duplicate", False))
                for slot_idx in range(batch_size)
            ]
            counted_slots = [slot_idx for slot_idx in range(batch_size) if not duplicate_flags[slot_idx]]

            episode_summaries = []
            for slot_idx in counted_slots:
                episode_summaries.append(
                    {
                        "episode_index": episode_numbers[slot_idx],
                        "episode_slot_index": slot_idx,
                        "episode_id": episode_ids[slot_idx],
                        "task_type": task_types[slot_idx],
                        "task_description": task_descriptions[slot_idx],
                        "steps": step_counts[slot_idx],
                        "reward": self.episode_rewards[slot_idx],
                        "success": success_flags[slot_idx],
                        "done": done_flags[slot_idx],
                        "active_strategic_node_id": active_strategic_node_ids[slot_idx],
                        "active_strategic_node_summary": strategic_selection_summaries[slot_idx],
                        "memory_retrievals": (
                            self._skill_retrievals[slot_idx]
                            if slot_idx < len(self._skill_retrievals)
                            else 0
                        ),
                        "memory_retrievals_refused_budget": (
                            self._skill_refused_budget[slot_idx]
                            if slot_idx < len(self._skill_refused_budget)
                            else 0
                        ),
                    }
                )

            counted_rewards = [self.episode_rewards[i] for i in counted_slots]
            counted_steps = [step_counts[i] for i in counted_slots]
            counted_success = [success_flags[i] for i in counted_slots]

            summary = {
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "mode": self.mode,
                "batch_size": batch_size,
                "max_steps": self.max_steps,
                "episodes": episode_summaries,
                "mean_reward": float(np.mean(counted_rewards)) if counted_rewards else 0.0,
                "mean_steps": float(np.mean(counted_steps)) if counted_steps else 0.0,
                "success_rate": float(np.mean(counted_success)) if counted_success else 0.0,
                "duplicate_slots": batch_size - len(counted_slots),
                "formation": formation_summary,
                "pruning": pruning_summary,
                "sleep_consolidation": sleep_summary,
                "sleep_bootstrap_tactical_min": self.sleep_bootstrap_tactical_min,
            }

            self._section_index += 1
            self._episodes_completed_cumulative += int(sum(done_flags[i] for i in counted_slots))
            self._report_metrics(
                {
                    "episode/mean_reward": summary["mean_reward"],
                    "episode/mean_steps": summary["mean_steps"],
                    "episode/success_rate": summary["success_rate"],
                    "episode/completed": int(sum(done_flags)),
                    # Running total across the whole run -- rises toward the
                    # planned episode count (e.g. num_epochs * num_games) as
                    # sections progress, unlike episode/completed above which
                    # resets to this batch's count (<= batch_size) every call.
                    "episode/completed_cumulative": self._episodes_completed_cumulative,
                    "episode/formation_candidates": formation_summary.get("candidates", 0),
                    "episode/formation_approved": formation_summary.get("approved", 0),
                    "episode/tactical_pruned": pruning_summary.get("pruned", 0),
                    # §P2.6.1 instrument 3: monotonic section counter -- its
                    # jumps mark section boundaries when this run's scalars
                    # are compared against episode/mean_reward in the same
                    # TensorBoard panel.
                    "sawtooth/section_index": self._section_index,
                }
            )
            self._report_task_type_metrics(episode_summaries)
            self._report_formation_pipeline_metrics(formation_gate_stats, formation_summary)
            self._report_baseline_metrics()
            self._report_graph_snapshot_metrics(pruning_summary)
            self._report_strategic_layer_metrics()
            self._report_sleep_consolidation_metrics(sleep_summary)

            summary_path = self.local_cache_dir / "episode_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

            # Append one JSON line per individual episode to a single,
            # run-lifetime file (never overwritten, never split into
            # per-episode directories) so it's trivial to check how many
            # episodes actually ran: `wc -l episodes.jsonl`.
            episodes_jsonl_path = self.local_cache_dir / "episodes.jsonl"
            with open(episodes_jsonl_path, "a", encoding="utf-8") as f:
                for episode in episode_summaries:
                    record = dict(episode)
                    record["run_id"] = self.run_id
                    record["experiment_name"] = self.experiment_name
                    record["global_step"] = self.current_step
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            return summary
        finally:
            self._close_tensorboard_writer()

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Serialize the cumulative, run-lifetime state that would otherwise
        reset to zero every time a fresh EpisodeRunner is constructed (each
        run_*.py invocation builds exactly one). The skill graph itself
        needs no separate snapshot here -- MemoryService persists every
        mutation straight to its sqlite db_path, so pointing a resumed run
        at the same db_path already recovers the memory side; this method
        only covers the runner's own progress/telemetry counters.
        """
        return {
            "episode_counter": self._episode_counter,
            "episodes_completed_cumulative": self._episodes_completed_cumulative,
            "section_index": self._section_index,
            "task_type_success_counts": dict(self._task_type_success_counts),
            "task_type_total_counts": dict(self._task_type_total_counts),
            "task_type_length_success": {
                k: list(v) for k, v in self._task_type_length_success.items()
            },
            "task_type_length_failure": {
                k: list(v) for k, v in self._task_type_length_failure.items()
            },
            "strategic_selection_counts": dict(self._strategic_selection_counts),
            "cumulative_nodes_created": self._cumulative_nodes_created,
            "cumulative_pruned_count": self._cumulative_pruned_count,
            "cumulative_pruned_by_task_type": dict(self._cumulative_pruned_by_task_type),
            "cumulative_sleep_action_counts": dict(self._cumulative_sleep_action_counts),
        }

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore counters saved by get_checkpoint_state(). Call this once,
        right after constructing a fresh EpisodeRunner and before the first
        run() call, when resuming an interrupted experiment."""
        self._episode_counter = int(state.get("episode_counter", self._episode_counter))
        self._episodes_completed_cumulative = int(
            state.get("episodes_completed_cumulative", self._episodes_completed_cumulative)
        )
        self._section_index = int(state.get("section_index", self._section_index))
        self._task_type_success_counts = dict(state.get("task_type_success_counts") or {})
        self._task_type_total_counts = dict(state.get("task_type_total_counts") or {})
        self._task_type_length_success = {
            k: list(v) for k, v in (state.get("task_type_length_success") or {}).items()
        }
        self._task_type_length_failure = {
            k: list(v) for k, v in (state.get("task_type_length_failure") or {}).items()
        }
        self._strategic_selection_counts = dict(state.get("strategic_selection_counts") or {})
        self._cumulative_nodes_created = int(state.get("cumulative_nodes_created", 0))
        self._cumulative_pruned_count = int(state.get("cumulative_pruned_count", 0))
        self._cumulative_pruned_by_task_type = dict(state.get("cumulative_pruned_by_task_type") or {})
        self._cumulative_sleep_action_counts = dict(state.get("cumulative_sleep_action_counts") or {})

    def close(self) -> None:
        """Release the environment adapter's resources.

        Callers that invoke `run()` repeatedly on the same instance (e.g. an
        outer num_sections loop) must call this once after the whole loop
        finishes, not after each `run()` -- closing the env adapter mid-loop
        forces it to lazily rebuild on the next reset(), which for ALFWorld
        means a brand-new env reseeded from scratch (losing the shuffled
        game-cycling position and effectively replaying the same game).
        """
        try:
            self.env_adapter.close()
        except Exception:
            logger.exception("Failed to close episode environment adapter")

    def _act_with_retry(
        self,
        *,
        agent: BaseAgent,
        observation: str,
        history_messages: List[Dict[str, Any]],
        first_step: bool,
        active_strategic_node_id: Optional[str],
        current_step: int,
        admissible_commands: Optional[Sequence[str]] = None,
        skill_budget_remaining: Optional[float] = None,
    ) -> AgentDecision:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                decision = agent.act(
                    observation=observation,
                    history_messages=history_messages,
                    first_step=first_step,
                    active_strategic_node_id=active_strategic_node_id,
                    current_step=current_step,
                    admissible_commands=admissible_commands or [],
                    skill_budget_remaining=skill_budget_remaining,
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

    def _auto_inject_memory(
        self,
        *,
        history: EpisodeHistory,
        observation: str,
        task_description: str,
        task_type: str,
        episode_id: str,
        active_strategic_node_id: Optional[str],
        slot_idx: Optional[int],
    ) -> Optional[MemoryRetrievalResult]:
        """Retrieve once, up front, and append the result as if the agent had
        asked for it.

        For single-step benchmarks (BCB: max_steps=1) agent-chosen retrieval is
        close to meaningless -- the agent has one decision point, no observation
        history, and no failed attempt to be uncertain about, so "should I
        retrieve?" carries no information. In practice it simply never asked
        (measured retrieval rate: 0), which quietly reduced the memory arm to
        the barebone baseline plus a longer prompt.

        Deliberately routed through the SAME MemoryRetrievalSkill call, budget,
        and counters as an agent-issued retrieval, so episodes.jsonl's
        memory_retrievals stays comparable across both modes and this shows up
        as a measurable ablation rather than a hidden behaviour change.
        """
        if self.memory_retrieval_skill is None:
            return None
        if (
            self.skill_budget_per_episode is not None
            and slot_idx is not None
            and slot_idx < len(self._skill_budget_remaining)
            and self._skill_budget_remaining[slot_idx] < 1
        ):
            return None

        try:
            retrieval_result = self.memory_retrieval_skill.retrieve(
                task_description=task_description,
                observation=observation,
                history_messages=self._history_to_messages(history),
                task_type=task_type,
                episode_id=episode_id,
                active_strategic_node_id=active_strategic_node_id,
                current_step=self.current_step,
                query_override=None,
            )
        except Exception as exc:
            logger.warning("Auto-injected memory retrieval failed: %s", exc)
            return None

        history.append_message(retrieval_result.to_tool_message(skill_name="memory_retrieval"))
        if slot_idx is not None and slot_idx < len(self._skill_budget_remaining):
            self._skill_budget_remaining[slot_idx] -= 1
        if slot_idx is not None and slot_idx < len(self._skill_retrievals):
            self._skill_retrievals[slot_idx] += 1
        logger.info(
            "Auto-injected memory (single-step mode): episode_id=%s selected=%s",
            episode_id,
            len(retrieval_result.selected_memories),
        )
        return retrieval_result

    def _resolve_agent_turn(
        self,
        *,
        agent: BaseAgent,
        observation: str,
        history: EpisodeHistory,
        first_step: bool,
        task_description: str,
        task_type: str,
        episode_id: str,
        active_strategic_node_id: Optional[str],
        admissible_commands: Optional[Sequence[str]] = None,
        slot_idx: Optional[int] = None,
    ) -> tuple[str, Optional[MemoryRetrievalResult]]:
        latest_retrieval_result: Optional[MemoryRetrievalResult] = None

        skill_budget_remaining: Optional[int] = None
        if (
            self.skill_budget_per_episode is not None
            and slot_idx is not None
            and slot_idx < len(self._skill_budget_remaining)
        ):
            skill_budget_remaining = self._skill_budget_remaining[slot_idx]

        if self.auto_inject_memory and first_step:
            injected = self._auto_inject_memory(
                history=history,
                observation=observation,
                task_description=task_description,
                task_type=task_type,
                episode_id=episode_id,
                active_strategic_node_id=active_strategic_node_id,
                slot_idx=slot_idx,
            )
            if injected is not None:
                latest_retrieval_result = injected
                if (
                    slot_idx is not None
                    and slot_idx < len(self._skill_budget_remaining)
                ):
                    skill_budget_remaining = self._skill_budget_remaining[slot_idx]

        for _ in range(self.max_skill_invocations + 1):
            history_messages = self._history_to_messages(history)
            decision = self._act_with_retry(
                agent=agent,
                observation=observation,
                history_messages=history_messages,
                first_step=first_step,
                active_strategic_node_id=active_strategic_node_id,
                current_step=self.current_step,
                admissible_commands=admissible_commands,
                skill_budget_remaining=skill_budget_remaining,
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

                if (
                    self.skill_budget_per_episode is not None
                    and slot_idx is not None
                    and slot_idx < len(self._skill_budget_remaining)
                    and self._skill_budget_remaining[slot_idx] < 1
                ):
                    if slot_idx is not None and slot_idx < len(self._skill_refused_budget):
                        self._skill_refused_budget[slot_idx] += 1
                    # Logged explicitly: this path returns without calling
                    # retrieve(), so it otherwise leaves no trace and looks
                    # identical to the agent never asking.
                    logger.info(
                        "Memory retrieval refused (budget exhausted): episode_id=%s step=%s "
                        "budget=%s",
                        episode_id,
                        self.current_step,
                        self.skill_budget_per_episode,
                    )
                    history.append_message(
                        {
                            "role": "tool",
                            "name": decision.skill_name or "memory_retrieval",
                            "content": (
                                f"Skill budget exhausted for this episode "
                                f"({self.skill_budget_per_episode} memory_retrieval call(s) already used). "
                                "Choose a direct environment action instead for the rest of the episode."
                            ),
                        }
                    )
                    continue

                if slot_idx is not None and slot_idx < len(self._skill_budget_remaining):
                    self._skill_budget_remaining[slot_idx] -= 1
                    skill_budget_remaining = self._skill_budget_remaining[slot_idx]

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
                    if slot_idx is not None and slot_idx < len(self._skill_retrievals):
                        self._skill_retrievals[slot_idx] += 1
                    history.append_message(
                        retrieval_result.to_tool_message(skill_name=decision.skill_name)
                    )
                    if hasattr(agent, "record_memory_retrieval"):
                        try:
                            agent.record_memory_retrieval(retrieval_result)
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
            self.max_skill_invocations,
        )
        return "look", latest_retrieval_result

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

    def _next_episode_number(self) -> int:
        self._episode_counter += 1
        return self._episode_counter

    def _report_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = dict(metrics)
        self.metrics_history.append(payload)
        self._append_metrics_jsonl(payload)
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

    def _append_metrics_jsonl(self, payload: Dict[str, Any]) -> None:
        """Append one JSON line per _report_metrics call to a single,
        run-lifetime file -- metrics_history is in-memory only (lost on
        process exit) and the TensorBoard event file isn't easy to load
        into pandas/duckdb for ad hoc analysis, so this is the structured,
        durable record. Mirrors the episodes.jsonl pattern.
        """
        record = dict(payload)
        record["global_step"] = self.current_step
        record["run_id"] = self.run_id
        try:
            metrics_jsonl_path = self.local_cache_dir / "metrics.jsonl"
            with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.debug("Failed to append metrics.jsonl", exc_info=True)

    def _seed_known_task_types(self) -> None:
        """Seed per-task-type metrics at zero for the env adapter's fixed taxonomy.

        Ensures per-task-type dashboards show every known type from the
        start of a run instead of only whatever types the early (possibly
        small) batches happen to sample. Delegates to
        `env_adapter.known_task_types()` so this stays benchmark-neutral --
        adapters without a fixed taxonomy (the base default) contribute
        nothing here.
        """
        known_task_types = []
        try:
            known_task_types = self.env_adapter.known_task_types()
        except Exception:
            logger.debug("Failed to read known_task_types from env_adapter", exc_info=True)
        for task_type in known_task_types:
            self._task_type_total_counts.setdefault(task_type, 0)
            self._task_type_success_counts.setdefault(task_type, 0)
            self._task_type_length_success.setdefault(task_type, [])
            self._task_type_length_failure.setdefault(task_type, [])

    def _report_task_type_metrics(self, episode_summaries: List[Dict[str, Any]]) -> None:
        """Per-task-type success rate and episode length at success vs. failure.

        Aggregate SR hides collapse on hard task types, so this reports the
        cumulative running per-type SR (updated with this batch's episodes)
        rather than just an overall number, plus whether successes are
        getting shorter over training (a sign of genuine skill reuse).
        """
        for episode in episode_summaries:
            task_type = episode.get("task_type") or "unknown"
            success = bool(episode.get("success"))
            steps = int(episode.get("steps") or 0)

            self._task_type_total_counts[task_type] = (
                self._task_type_total_counts.get(task_type, 0) + 1
            )
            if success:
                self._task_type_success_counts[task_type] = (
                    self._task_type_success_counts.get(task_type, 0) + 1
                )
                self._task_type_length_success.setdefault(task_type, []).append(steps)
            else:
                self._task_type_length_failure.setdefault(task_type, []).append(steps)

        metrics: Dict[str, Any] = {}
        for task_type, total in self._task_type_total_counts.items():
            metrics[f"task_type/episode_count/{task_type}"] = total
            if total <= 0:
                continue
            successes = self._task_type_success_counts.get(task_type, 0)
            metrics[f"task_type/success_rate/{task_type}"] = float(successes) / float(total)

            success_lengths = self._task_type_length_success.get(task_type) or []
            failure_lengths = self._task_type_length_failure.get(task_type) or []
            if success_lengths:
                metrics[f"task_type/mean_length_success/{task_type}"] = float(
                    np.mean(success_lengths)
                )
            if failure_lengths:
                metrics[f"task_type/mean_length_failure/{task_type}"] = float(
                    np.mean(failure_lengths)
                )

        if metrics:
            self._report_metrics(metrics)

    def _report_formation_pipeline_metrics(
        self,
        formation_gate_stats: Dict[str, int],
        formation_summary: Dict[str, Any],
    ) -> None:
        """Stage-1 admission rate (overall / by outcome / by step position)
        and the storage rate of queued candidates. The by-outcome split
        should diverge (if it doesn't, b(t_k) isn't discriminating); the
        by-position split checks for the recency-skew failure mode. There
        is no LLM judge anymore -- storage_rate should sit at ~1.0 (every
        stage-1-admitted candidate is stored); a persistent drop below 1.0
        would flag a bug, not a judgment call."""

        def _rate(admitted_key: str, total_key: str) -> Optional[float]:
            total = formation_gate_stats.get(total_key, 0)
            if not total:
                return None
            return float(formation_gate_stats.get(admitted_key, 0)) / float(total)

        metrics: Dict[str, Any] = {}
        admission_rate = _rate("admitted_steps", "total_steps")
        if admission_rate is not None:
            metrics["formation/stage1_admission_rate"] = admission_rate
        for outcome in ("success", "failure"):
            rate = _rate(f"admitted_{outcome}_steps", f"total_{outcome}_steps")
            if rate is not None:
                metrics[f"formation/stage1_admission_rate_{outcome}"] = rate
        for position in ("early", "mid", "late"):
            rate = _rate(f"admitted_{position}_steps", f"total_{position}_steps")
            if rate is not None:
                metrics[f"formation/stage1_admission_rate_{position}"] = rate

        candidates = formation_summary.get("candidates", 0) or 0
        approved = formation_summary.get("approved", 0) or 0
        if candidates:
            metrics["formation/storage_rate"] = float(approved) / float(candidates)

        created = len(formation_summary.get("created_nodes") or [])
        self._cumulative_nodes_created += created
        metrics["formation/new_nodes"] = created
        metrics["formation/new_nodes_cumulative"] = self._cumulative_nodes_created

        if metrics:
            self._report_metrics(metrics)

    def _report_baseline_metrics(self) -> None:
        """Per-task-type advantage baselines b(t_k) / b^Omega(t_k) (spec §2.7).

        Reported every batch so convergence (or continued drift late in
        training) is visible over time, not just the final value.
        """
        graph = getattr(self.memory_service, "graph", None)
        if graph is None:
            return
        metrics: Dict[str, Any] = {}
        for task_type, value in (getattr(graph, "baseline_tactical", None) or {}).items():
            metrics[f"baseline/tactical/{task_type}"] = float(value)
        for task_type, value in (getattr(graph, "baseline_strategic", None) or {}).items():
            metrics[f"baseline/strategic/{task_type}"] = float(value)
        if metrics:
            self._report_metrics(metrics)

    def _report_graph_snapshot_metrics(self, pruning_summary: Dict[str, Any]) -> None:
        """Tactical graph size, decay-rate distribution, and pruning counts.

        `graph/decay_rate_mean/min/max` are computed over the SURVIVING
        tactical nodes only -- a pruned node stops contributing to them the
        moment it's removed (`SkillGraph.remove` pops it from `graph.nodes`,
        and this method reads `graph.nodes_at_depth(2)` after pruning already
        ran this batch, §10). That's correct for "current live graph," but it
        creates the same convergence-vs-survivorship ambiguity the §P2.6.1
        n_omega overlay was built to resolve for the strategic tier: a
        falling decay_rate_mean can mean the tactical layer is genuinely
        getting better, OR it can just mean pruning keeps removing the worst
        (highest-decay-rate) nodes, leaving only survivors. Always read
        decay_rate_mean next to tactical_node_count (falling mean + falling
        count = likely survivorship; falling mean + stable/rising count =
        likely genuine improvement) -- graph/pruned_fraction_of_created
        below is a single-scalar version of that same check.

        Pruned-count-by-task-type is tracked cumulatively to check for
        differential starvation across easy/hard task types.
        """
        graph = getattr(self.memory_service, "graph", None)
        if graph is None:
            return
        tactical_nodes = graph.nodes_at_depth(2) if hasattr(graph, "nodes_at_depth") else []
        metrics: Dict[str, Any] = {"graph/tactical_node_count": len(tactical_nodes)}

        decay_rates = [float(node.decay_rate) for node in tactical_nodes]
        if decay_rates:
            metrics["graph/decay_rate_mean"] = float(np.mean(decay_rates))
            metrics["graph/decay_rate_min"] = float(np.min(decay_rates))
            metrics["graph/decay_rate_max"] = float(np.max(decay_rates))

        pruned_this_epoch = int(pruning_summary.get("pruned", 0) or 0)
        self._cumulative_pruned_count += pruned_this_epoch
        metrics["graph/pruned_this_epoch"] = pruned_this_epoch
        metrics["graph/pruned_cumulative"] = self._cumulative_pruned_count

        # Survivorship-bias check: what fraction of every tactical node ever
        # formed has since been pruned. High and rising alongside a falling
        # decay_rate_mean is the tell that the mean's improvement is mostly
        # attrition, not genuine skill quality gains.
        if self._cumulative_nodes_created:
            metrics["graph/pruned_fraction_of_created"] = float(
                self._cumulative_pruned_count
            ) / float(self._cumulative_nodes_created)

        pruned_by_task_type = pruning_summary.get("pruned_by_task_type") or {}
        for task_type, count in pruned_by_task_type.items():
            self._cumulative_pruned_by_task_type[task_type] = (
                self._cumulative_pruned_by_task_type.get(task_type, 0) + count
            )
        for task_type, count in self._cumulative_pruned_by_task_type.items():
            metrics[f"graph/pruned_cumulative/{task_type}"] = count

        self._report_metrics(metrics)

    def _report_strategic_layer_metrics(self) -> None:
        """Strategic scaffold count, per-scaffold Q_omega, cross-scaffold
        spread, and selection frequency.

        Near-zero Q_omega variance across scaffolds for the same task type
        means the scaffolds aren't functionally differentiated even if
        individually non-zero; selection frequency flags one scaffold
        dominating every episode regardless of task type.
        """
        graph = getattr(self.memory_service, "graph", None)
        if graph is None:
            return
        scaffolds = graph.nodes_at_depth(1) if hasattr(graph, "nodes_at_depth") else []
        metrics: Dict[str, Any] = {"strategic/scaffold_count": len(scaffolds)}

        per_task_type_values: Dict[str, List[float]] = {}
        total_selections = sum(self._strategic_selection_counts.values())
        selection_counts: List[int] = []
        for scaffold in scaffolds:
            short_id = str(scaffold.id)[:8]
            for task_type, value in (scaffold.Q_omega or {}).items():
                metrics[f"strategic/q_omega/{short_id}/{task_type}"] = float(value)
                per_task_type_values.setdefault(task_type, []).append(float(value))

            # §P2.6.1 instrument 1: n_omega per (scaffold, task_type) next to
            # Q_omega/variance on the same step axis -- a falling value or
            # variance is convergence if n_omega keeps rising, but starvation
            # (deranked option never re-accrues experience) if n_omega is flat.
            for task_type, count in (scaffold.n_omega or {}).items():
                metrics[f"strategic/n_omega/{short_id}/{task_type}"] = int(count)

            selection_count = self._strategic_selection_counts.get(scaffold.id, 0)
            selection_counts.append(selection_count)
            metrics[f"strategic/selection_count/{short_id}"] = selection_count
            if total_selections:
                metrics[f"strategic/selection_fraction/{short_id}"] = (
                    float(selection_count) / float(total_selections)
                )

        for task_type, values in per_task_type_values.items():
            if len(values) >= 2:
                metrics[f"strategic/q_omega_variance/{task_type}"] = float(np.var(values))

        # §P2.6.1 instrument 5: rich-get-richer selection imbalance audit --
        # a single ratio to watch for one scaffold dominating while a newer
        # one is starved (observed live: ~6x spread across 4 scaffolds).
        nonzero_counts = [count for count in selection_counts if count > 0]
        if nonzero_counts:
            metrics["strategic/selection_count_spread_ratio"] = float(max(nonzero_counts)) / float(
                min(nonzero_counts)
            )

        self._report_metrics(metrics)
        self._append_strategic_scaffolds_jsonl(scaffolds)

    def _append_strategic_scaffolds_jsonl(self, scaffolds: List[Any]) -> None:
        """One JSON line per strategic scaffold, per batch -- id, content
        (the scaffold's actual strategy text), per-task-type advantage
        (Q_omega/n_omega), and selection count.

        A structured companion to metrics.jsonl: that file only keeps
        Q_omega/n_omega keyed by an 8-char truncated id (kept short for TB
        tag hygiene), so it can't be joined back to a specific scaffold's
        content or full id. This keeps both, for reading what a scaffold's
        strategy text says right next to how it's performing.
        """
        if not scaffolds:
            return
        path = self.local_cache_dir / "strategic_scaffolds.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for scaffold in scaffolds:
                try:
                    summary = self.memory_service.get_representation(scaffold.id).content
                except Exception:
                    logger.debug(
                        "Failed to fetch representation for scaffold %s", scaffold.id, exc_info=True
                    )
                    summary = None
                record = {
                    "run_id": self.run_id,
                    "global_step": self.current_step,
                    "section_index": self._section_index,
                    "scaffold_id": scaffold.id,
                    "task_type_dominant": scaffold.task_type_dominant,
                    "t_create": scaffold.t_create,
                    "summary": summary,
                    "q_omega": dict(scaffold.Q_omega or {}),
                    "n_omega": dict(scaffold.n_omega or {}),
                    "selection_count": self._strategic_selection_counts.get(scaffold.id, 0),
                    "evidence_ids": list(scaffold.evidence_ids or []),
                    # evidence_ids covers POSITIVE evidence only. The failure
                    # traces are in-memory and destroyed on a successful
                    # revision, and the revised steps read as ordinary
                    # instructions, so this count is the only durable signal
                    # that reflection reached this scaffold.
                    "reflections_absorbed": int(
                        getattr(scaffold, "reflections_absorbed", 0) or 0
                    ),
                    "pending_failure_traces": len(
                        self.memory_service.graph.failure_buffer.get(scaffold.id, [])
                    ),
                }
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _report_sleep_consolidation_metrics(
        self, sleep_summary: Optional[Dict[str, Any]]
    ) -> None:
        """Sleep-consolidation trigger/eligibility/clustering/action counts.

        Only reports when consolidation actually ran this batch (fires
        rarely, gated by n_sleep).
        """
        if not sleep_summary or not sleep_summary.get("consolidation_ran"):
            return
        metrics: Dict[str, Any] = {
            "sleep/trigger_step": sleep_summary.get("trigger_step"),
            "sleep/unconsolidated_count": sleep_summary.get("unconsolidated_count"),
            "sleep/eligible_count": sleep_summary.get("eligible_count"),
            # cluster_count is the RAW number of clusters k-means formed;
            # num_results is how many of those got a successful LLM decision
            # and were actually acted on. cluster_count > num_results means
            # some clusters' decisions failed and were silently skipped --
            # see cluster_decision_failed_count below, not a sign that
            # clustering itself collapsed to fewer clusters.
            "sleep/cluster_count": sleep_summary.get("cluster_count"),
            "sleep/num_results": sleep_summary.get("num_results"),
            "sleep/cluster_decision_failed_count": sleep_summary.get("cluster_decision_failed_count"),
        }
        cluster_sizes = sleep_summary.get("cluster_sizes") or []
        if cluster_sizes:
            metrics["sleep/cluster_size_min"] = min(cluster_sizes)
            metrics["sleep/cluster_size_max"] = max(cluster_sizes)
            metrics["sleep/cluster_size_mean"] = float(np.mean(cluster_sizes))
            total = sum(cluster_sizes)
            if total:
                # Flags heavy skew, e.g. one cluster holding 90% of nodes.
                metrics["sleep/cluster_size_max_fraction"] = max(cluster_sizes) / float(total)

        db_score = sleep_summary.get("cluster_davies_bouldin")
        if db_score is not None:
            metrics["sleep/cluster_davies_bouldin"] = db_score

        action_counts = sleep_summary.get("action_counts") or {}
        for action_name, count in action_counts.items():
            metrics[f"sleep/action_{action_name}"] = count
            # §P2.6.1 instrument 4: cumulative per-action count over the
            # whole run. Sleep events are rare and per-event action_counts
            # are easy to miss between them -- a run-lifetime "has absorb
            # ever fired" answer should be readable off one flat-vs-rising
            # scalar, not require correlating snapshots across events.
            self._cumulative_sleep_action_counts[action_name] = (
                self._cumulative_sleep_action_counts.get(action_name, 0) + int(count)
            )
        for action_name, cumulative_count in self._cumulative_sleep_action_counts.items():
            metrics[f"sleep/action_{action_name}_cumulative"] = cumulative_count

        # §P2.6.1 instrument 3: marks sleep-consolidation events when
        # overlaid against episode/mean_reward on the shared step axis.
        metrics["sawtooth/sleep_consolidation_marker"] = 1

        self._report_metrics({key: value for key, value in metrics.items() if value is not None})

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

    def _update_episode_tactical_q(
        self,
        *,
        task_types: List[str],
        reward_histories: List[List[float]],
        active_tactical_visits: List[List[Optional[str]]],
        dirty_nodes: Dict[str, Any],
    ) -> None:
        """Monte Carlo return-to-go tactical Q update, committed at episode end (spec §3.2).

        No bootstrap. For every step of the buffered trajectory, G_t is the
        discounted return-to-go and the update target is the advantage
        A_t = G_t - b(t_k), where b(t_k) is the per-task-type baseline read
        before it is updated with this episode's discounted return (§3.1, §4.1).
        """
        graph = getattr(self.memory_service, "graph", None)
        if graph is None:
            return

        gamma = float(getattr(self.memory_config, "gamma", 0.95))
        alpha = float(getattr(self.memory_config, "alpha", 0.1))
        lambda_shrink = float(getattr(self.memory_config, "lambda_shrink", 10.0))

        for slot_idx, rewards in enumerate(reward_histories):
            visits = active_tactical_visits[slot_idx] if slot_idx < len(active_tactical_visits) else []
            step_count = min(len(rewards), len(visits))
            if step_count <= 0:
                continue

            task_type = task_types[slot_idx]
            returns_to_go = compute_mc_return_to_go(rewards[:step_count], gamma=gamma)
            baseline = graph.get_tactical_baseline(task_type)

            for step_idx in range(step_count):
                node_id = visits[step_idx]
                if node_id is None:
                    continue
                node = graph.nodes.get(node_id)
                if node is None or not getattr(node, "is_tactical", False):
                    continue

                current_value = float((node.Q or {}).get(task_type, 0.0))
                advantage = compute_advantage(returns_to_go[step_idx], baseline)
                updated_value = apply_q_update(
                    current_value,
                    advantage - current_value,
                    alpha=alpha,
                )
                node.Q[task_type] = updated_value
                node.n[task_type] = int(node.n.get(task_type, 0) or 0) + 1
                node.refresh_task_type_dominant()
                node.last_accessed_step = self.current_step
                if hasattr(graph, "refresh_decay_rate"):
                    graph.refresh_decay_rate(node)
                else:
                    node.recompute_decay_rate(
                        lambda_base=float(getattr(self.memory_config, "lambda_base", 0.0) or 0.0),
                        epsilon=float(getattr(self.memory_config, "epsilon_decay", 0.01)),
                        lambda_shrink=lambda_shrink,
                    )

                log_event(
                    logger,
                    "tactical_q.update",
                    node_id=node_id,
                    task_type=task_type,
                    return_to_go=returns_to_go[step_idx],
                    baseline=baseline,
                    advantage=advantage,
                    current_value=current_value,
                    updated_value=updated_value,
                    visit_count=node.n.get(task_type, 0),
                )
                dirty_nodes[node.id] = node

                self._report_metrics(
                    {
                        "episode/tactical_advantage": advantage,
                        "episode/tactical_q": float(updated_value),
                        "episode/tactical_salience": float(
                            get_q_salience(node, lambda_shrink=lambda_shrink)
                        ),
                    }
                )

            graph.update_tactical_baseline(task_type, returns_to_go[0])

    def _update_episode_q_omega(
        self,
        *,
        task_types: List[str],
        reward_histories: List[List[float]],
        step_infos: List[Dict[str, Any]],
        active_strategic_node_ids: List[Optional[str]],
        dirty_nodes: Dict[str, Any],
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
                # Strategic advantage vs the per-task-type baseline b^Omega(t_k),
                # read before this episode's return updates it (spec §3.8).
                baseline = self.memory_service.graph.get_strategic_baseline(task_type)
                advantage = compute_advantage(episode_return, baseline)
                updated_value = apply_q_update(
                    current_value,
                    advantage - current_value,
                    alpha=alpha_omega,
                )
                node.Q_omega[task_type] = updated_value
                node.n_omega[task_type] = int(node.n_omega.get(task_type, 0) or 0) + 1
                node.refresh_task_type_dominant()
                self.memory_service.graph.update_strategic_baseline(task_type, episode_return)
                log_event(
                    logger,
                    "strategic_q.update",
                    node_id=node.id,
                    task_type=task_type,
                    episode_return=episode_return,
                    baseline=baseline,
                    advantage=advantage,
                    current_value=current_value,
                    updated_value=updated_value,
                    visit_count=node.n_omega.get(task_type, 0),
                )

            dirty_nodes[node.id] = node

            short_id = str(node.id)[:8]
            self._report_metrics(
                {
                    "episode/omega_return": episode_return,
                    "episode/omega_q": get_q_omega_salience(
                        node,
                        lambda_shrink=float(getattr(self.memory_config, "lambda_shrink", 10.0)),
                    ),
                    # §P2.6.1 instrument 2: raw G^Omega and the baseline it's
                    # measured against, scoped per (scaffold, task_type) so
                    # they can be read alongside the stored advantage
                    # (strategic/q_omega/.../...) without colliding across
                    # scaffolds/slots in the same batch -- a declining
                    # advantage is benign if the raw return is flat (the
                    # baseline caught up) and concerning if both fall.
                    f"strategic/g_omega_raw/{short_id}/{task_type}": episode_return,
                    f"strategic/b_omega/{task_type}": baseline,
                    f"strategic/advantage/{short_id}/{task_type}": advantage,
                }
            )

    def _queue_failed_episode_reflections(
        self,
        *,
        task_descriptions: List[str],
        success_flags: List[bool],
        active_strategic_node_ids: List[Optional[str]],
    ) -> None:
        """Buffer failed episodes onto their active scaffold's failure buffer.

        This is the reflection channel's capture side: a condensed trace of
        every failed episode with an active strategic scaffold is appended
        in-memory (`graph.record_failure`) -- unconditionally, no solvability
        gate. Sleep consolidation Pass 2 later consumes these as negative
        evidence and flushes them only once a revision succeeds -- nothing
        here touches SQLite.
        """
        graph = getattr(self.memory_service, "graph", None)
        if graph is None:
            return
        for slot_idx, success in enumerate(success_flags):
            if success:
                continue
            node_id = (
                active_strategic_node_ids[slot_idx]
                if slot_idx < len(active_strategic_node_ids)
                else None
            )
            if node_id is None:
                continue
            history = self.episode_histories[slot_idx]
            trace = (
                f"Task: {task_descriptions[slot_idx]}\n"
                f"{history.get_formatted_history()}\n"
                f"Outcome: failed (reward={self.episode_rewards[slot_idx]})"
            )
            graph.record_failure(node_id, trace)
            log_event(
                logger,
                "reflection.failure_recorded",
                scaffold_id=node_id,
                episode_slot_index=slot_idx,
                task_description=task_descriptions[slot_idx],
                reward=self.episode_rewards[slot_idx],
            )

    def _flush_dirty_nodes(self, dirty_nodes: Dict[str, Any]) -> None:
        """Batch-persist the episode's in-memory working set (spec §5.3).

        Called after pruning, so a node removed by decay-based pruning this
        same episode is dropped from the flush instead of being written
        back and resurrecting a row that was just deleted.
        """
        if not dirty_nodes:
            return
        graph = getattr(self.memory_service, "graph", None)
        if graph is None:
            return
        surviving = [
            node for node_id, node in dirty_nodes.items() if graph.has_node(node_id)
        ]
        if not surviving:
            return
        if hasattr(self.memory_service, "persist_nodes"):
            self.memory_service.persist_nodes(surviving)
        else:
            for node in surviving:
                self.memory_service.persist_node_state(node)

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

        # Each benchmark's agent subclass can define its own domain-flavored
        # scaffold-selection prompts (e.g. BCBAgent/LLBAgent) via the
        # strategic_selection_system_prompt/_user_prompt class attributes
        # (memrl/agent/custom_agent.py); fall back to ALFWorld's original
        # wording for any agent that doesn't define them.
        strategic_selection_system_prompt = getattr(
            self.agent, "strategic_selection_system_prompt", None
        ) or agent_prompts.STRATEGIC_SELECTION_SYSTEM_PROMPT
        strategic_selection_user_prompt = getattr(
            self.agent, "strategic_selection_user_prompt", None
        ) or agent_prompts.STRATEGIC_SELECTION_USER_PROMPT

        prompt_messages = [
            {"role": "system", "content": strategic_selection_system_prompt},
            {
                "role": "user",
                "content": strategic_selection_user_prompt.format(
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
        advantage: Optional[float],
    ) -> bool:
        """Stage-1 advantage pre-filter (spec §4.1): A_t = G_t - b(t_k) > theta_adv."""
        if advantage is None:
            return False
        theta_adv = float(getattr(self.memory_config, "theta_adv", 0.0) or 0.0)
        if advantage <= theta_adv:
            log_event(
                logger,
                "tactical_formation.rejected",
                reason="advantage_below_theta_adv",
                advantage=advantage,
                theta_adv=theta_adv,
            )
            return False
        return True

    def _commit_pending_formations(self) -> Dict[str, Any]:
        candidates_raw = list(self.pending_formations)
        self.pending_formations = []
        if not candidates_raw:
            return {"candidates": 0, "approved": 0, "created_nodes": [], "skipped": False}

        log_event(
            logger,
            "tactical_formation.start",
            raw_candidates=len(candidates_raw),
        )

        # No LLM judge anymore: the phase-1 advantage gate is the sole
        # formation decision. Every candidate that passes it is stored --
        # the LLM's only remaining role (tactical_summary_writer, below) is
        # summarizing it into a procedural memory, not deciding whether to
        # keep it.
        candidates = [
            TacticalFormationCandidate(**candidate)
            for candidate in candidates_raw
            if self._should_queue_tactical_candidate(
                advantage=float(candidate.get("advantage", 0.0) or 0.0),
            )
        ]
        log_event(
            logger,
            "tactical_formation.filtered",
            raw_candidates=len(candidates_raw),
            passed_candidates=len(candidates),
        )
        if not candidates:
            return {
                "candidates": len(candidates_raw),
                "approved": 0,
                "created_nodes": [],
                "skipped": True,
                "filtered": True,
            }

        created_nodes: List[str] = []
        for candidate in candidates:
            parent_id = candidate.active_strategic_node_id or self.memory_service.graph.root_id
            node_id = uuid4().hex
            evidence_ids = [candidate.candidate_id]
            if candidate.source_memory_id:
                evidence_ids.append(candidate.source_memory_id)

            # Persist the raw trace this skill was formed from into the
            # episodic bank, addressed by the same id evidence_ids points
            # to (spec §1, §5.4). The node only ever surfaces its LLM
            # summary at retrieval; this keeps the trace available for
            # inspection and future credit-assignment work (§11).
            self.memory_service.record_evidence(
                EpisodicRecord(
                    id=candidate.candidate_id,
                    task_type=candidate.task_type,
                    task_description=candidate.task_description,
                    episode_id=candidate.episode_id,
                    step_index=candidate.step_index,
                    observation=candidate.observation,
                    action=candidate.action,
                    reward=candidate.reward,
                    history=candidate.history,
                    retrieved_memories=candidate.retrieved_memories,
                    source_memory_id=candidate.source_memory_id,
                )
            )

            summary_content = candidate.fallback_summary()
            summary_writer = self.tactical_summary_writer
            if summary_writer is not None:
                try:
                    summary_draft = summary_writer.summarize_candidate(candidate)
                    summary_content = summary_writer.format_summary(summary_draft) or summary_content
                except Exception as exc:
                    logger.warning(
                        "Tactical summary generation failed; using fallback summary instead: %s",
                        exc,
                    )
            log_event(
                logger,
                "tactical_formation.decision",
                candidate_id=candidate.candidate_id,
                approved=True,
                summary=summary_content,
                parent_id=parent_id,
                source_memory_id=candidate.source_memory_id,
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
                # Seed Q from the advantage that admitted this candidate
                # (spec §4.1/§3.3) -- otherwise the node starts at salience
                # 0 despite already having known positive evidence, and
                # stays ineligible for sleep consolidation until some later
                # episode happens to retrieve and re-update it.
                initial_q={candidate.task_type: candidate.advantage},
                initial_n={candidate.task_type: 1},
            )
            created_nodes.append(node_id)

        return {
            "candidates": len(candidates_raw),
            "approved": len(created_nodes),
            "created_nodes": created_nodes,
            "skipped": False,
        }

    @staticmethod
    def _step_position_bucket(step_idx: int, step_count: int) -> str:
        """Bucket a 0-indexed step into early/mid/late thirds of its episode."""
        if step_count <= 1:
            return "mid"
        fraction = step_idx / float(step_count - 1)
        if fraction < 1.0 / 3.0:
            return "early"
        if fraction < 2.0 / 3.0:
            return "mid"
        return "late"

    @staticmethod
    def _new_formation_gate_stats() -> Dict[str, int]:
        keys = ["total", "admitted"]
        buckets = ["", "_success", "_failure", "_early", "_mid", "_late"]
        return {f"{key}{bucket}_steps": 0 for key in keys for bucket in buckets}

    def _queue_episode_tactical_candidates(
        self,
        *,
        reward_histories: List[List[float]],
        candidate_buffers: List[List[Dict[str, Any]]],
        success_flags: Optional[List[bool]] = None,
    ) -> Dict[str, int]:
        """Stage-1 advantage pre-filter, batched at episode end (spec §4.1).

        G_t is the MC return-to-go from backward discounted recursion.
        Admission is gated on the advantage against the per-task-type
        baseline b(t_k), not raw return: A_t = G_t - b(t_k) > theta_adv.
        This is the same baseline tracker used by the tactical Q update
        (§3.1) — read here before `_update_episode_tactical_q` updates it
        for this episode, so an episode is scored against history excluding
        itself. Callers must invoke this before `_update_episode_tactical_q`
        for the same episode.

        Returns admission-rate bookkeeping (total/admitted step counts,
        split by episode outcome and by within-episode step position) so
        the caller can report the Stage-1 admission rate and check whether
        it discriminates by outcome and isn't recency-skewed.
        """
        graph = getattr(self.memory_service, "graph", None)
        gamma = float(getattr(self.memory_config, "gamma", 0.95))
        stats = self._new_formation_gate_stats()
        for slot_idx, candidates in enumerate(candidate_buffers):
            if not candidates:
                continue

            rewards = reward_histories[slot_idx] if slot_idx < len(reward_histories) else []
            step_count = min(len(candidates), len(rewards))
            if step_count <= 0:
                continue

            task_type = candidates[0].get("task_type")
            baseline = graph.get_tactical_baseline(task_type) if graph is not None else 0.0
            returns_to_go = compute_mc_return_to_go(rewards[:step_count], gamma=gamma)
            success = (
                bool(success_flags[slot_idx])
                if success_flags is not None and slot_idx < len(success_flags)
                else None
            )
            outcome_bucket = None if success is None else ("success" if success else "failure")

            queued_count = 0
            for step_idx in range(step_count):
                candidate = dict(candidates[step_idx])
                candidate["advantage"] = compute_advantage(returns_to_go[step_idx], baseline)
                admitted = self._should_queue_tactical_candidate(advantage=candidate["advantage"])
                if admitted:
                    self.pending_formations.append(candidate)
                    queued_count += 1

                position_bucket = self._step_position_bucket(step_idx, step_count)
                stats["total_steps"] += 1
                stats[f"total_{position_bucket}_steps"] += 1
                if outcome_bucket is not None:
                    stats[f"total_{outcome_bucket}_steps"] += 1
                if admitted:
                    stats["admitted_steps"] += 1
                    stats[f"admitted_{position_bucket}_steps"] += 1
                    if outcome_bucket is not None:
                        stats[f"admitted_{outcome_bucket}_steps"] += 1

            log_event(
                logger,
                "tactical_formation.episode_backfill",
                episode_index=candidates[0].get("episode_index"),
                episode_id=candidates[0].get("episode_id"),
                step_count=step_count,
                queued_count=queued_count,
                baseline=baseline,
                propagated_return=returns_to_go[0] if returns_to_go else 0.0,
            )

        return stats

    def _prune_tactical_nodes(self) -> Dict[str, Any]:
        theta_prune = getattr(self.memory_config, "theta_prune", None)
        if theta_prune is None:
            return {"pruned": 0, "pruned_node_ids": [], "theta_prune": None, "pruned_by_task_type": {}}

        task_type_counts: Dict[str, int] = {}
        pruned_node_ids = self.memory_service.prune_tactical_nodes(
            current_step=self.current_step,
            theta_prune=float(theta_prune),
            task_type_counts_out=task_type_counts,
        )
        return {
            "pruned": len(pruned_node_ids),
            "pruned_node_ids": pruned_node_ids,
            "theta_prune": float(theta_prune),
            "pruned_by_task_type": task_type_counts,
        }
