"""Barebone LifelongBench (LLB) runner: plain multi-turn LLM agent, no memory.

Standalone by design -- does not import EpisodeRunner, MemoryService, or
anything under memrl.agent/memrl.skills/memrl.service. This is the true
"No-Memory" baseline, structured like the ALFWorld barebone runner (running
interaction history + current observation each turn) but with no
admissible-actions list, since LLB's DB/OS actions are free-form SQL/bash,
not a fixed command set.

Unlike ALFWorld's natively batched env, each LLB episode owns its own
Task/Session pair (and, for DB/OS, its own Docker container spun up inside
task_obj.reset()) -- there is no existing parallel-session primitive to
reuse safely here, so this runner processes episodes SEQUENTIALLY, one
Task/Session at a time, rather than in parallel batches like the ALFWorld/
BCB barebone runners. That's a deliberate simplification, not an oversight.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.envs.llb_episode_adapter import _latest_observation
from memrl.lifelongbench_eval.prompts import normalize_llb_action_directive
from memrl.lifelongbench_eval.task_wrappers import (
    ChatHistoryItem,
    Role,
    SampleStatus,
    Session,
    SessionEvaluationOutcome,
    build_task,
    sorted_sample_indices,
)
from memrl.providers.llm import OpenAILLM

from agent import BarebonLLBAgent

logger = logging.getLogger(__name__)


def setup_logging(name: str) -> Path:
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info("Logging configured. Log file: %s", log_path)
    return log_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Barebone LifelongBench (LLB) runner -- no memory, no tools.")
    p.add_argument("--task", type=str, required=True, choices=["db", "db_bench", "os", "os_interaction"])
    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the LLB dataset JSON (a dict keyed by sample_index), e.g. "
        "data/llb/os_interaction_train.json or data/llb/db_train.json.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap on number of tasks (default: all).")
    p.add_argument("--num_epochs", type=int, default=1, help="Number of full dataset replays.")
    p.add_argument("--max_round", type=int, default=15, help="Max agent turns per episode.")
    p.add_argument("--os_timeout", type=int, default=20, help="Per-command execution timeout (OS task only).")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--api_key", type=str, default=None, help="Falls back to OPENAI_API_KEY env var.")
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--output_dir", type=str, default=str(project_root / "results" / "barebone"))
    p.add_argument("--smoke", action="store_true", help="Run a single task, then exit.")
    return p.parse_args()


def run_one_episode(
    *,
    task: str,
    data_path: str,
    sample_id: str,
    max_round: int,
    os_timeout: int,
    agent: BarebonLLBAgent,
) -> dict:
    task_obj, task_name = build_task(
        task=task, data_file_path=data_path, max_round=max_round, os_timeout=os_timeout
    )
    session = Session(task_name=task_name, sample_index=sample_id)

    try:
        task_obj.reset(session)
    except Exception as exc:
        logger.error("LLB reset failed for sample_id=%s: %s", sample_id, exc, exc_info=True)
        return {"task_id": sample_id, "status": "RESET_ERROR", "error": str(exc), "pass": False, "steps": 0}

    observation = _latest_observation(session)
    agent.reset(task_description=observation)
    steps = 0

    try:
        while session.sample_status == SampleStatus.RUNNING:
            response = normalize_llb_action_directive(agent.act(observation), task)
            session.chat_history.inject(ChatHistoryItem(role=Role.AGENT, content=response))
            task_obj.interact(session)
            steps += 1
            observation = _latest_observation(session)
    except Exception as exc:
        logger.error("LLB interact failed for sample_id=%s: %s", sample_id, exc, exc_info=True)
        return {"task_id": sample_id, "status": "INTERACT_ERROR", "error": str(exc), "pass": False, "steps": steps}

    try:
        task_obj.complete(session)
        ok = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
    except Exception as exc:
        logger.error("LLB complete failed for sample_id=%s: %s", sample_id, exc, exc_info=True)
        ok = False
    finally:
        release = getattr(task_obj, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                logger.debug("task_obj.release() failed for sample_id=%s", sample_id, exc_info=True)

    return {
        "task_id": sample_id,
        "status": str(session.sample_status),
        "finish_reason": session.finish_reason,
        "pass": ok,
        "steps": steps,
    }


def main() -> None:
    args = parse_args()
    setup_logging("llb_barebone")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key: pass --api_key or set OPENAI_API_KEY.")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"exp_llb_{args.task}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_provider = OpenAILLM(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        default_temperature=args.temperature,
        default_max_tokens=args.max_tokens,
        token_log_dir=str(run_dir),
    )
    agent = BarebonLLBAgent(llm_provider, task=args.task)

    limit = 1 if args.smoke else args.limit
    sample_ids = sorted_sample_indices(args.data_path, limit)
    num_epochs = 1 if args.smoke else args.num_epochs
    logger.info(
        "Running %s epoch(s) over %s LLB %s task(s).",
        num_epochs, len(sample_ids), args.task,
    )

    episodes_path = run_dir / "episodes.jsonl"

    for epoch_idx in range(num_epochs):
        pass_count = 0
        for idx, sample_id in enumerate(sample_ids, start=1):
            result = run_one_episode(
                task=args.task,
                data_path=args.data_path,
                sample_id=sample_id,
                max_round=args.max_round,
                os_timeout=args.os_timeout,
                agent=agent,
            )
            pass_count += 1 if result["pass"] else 0

            with open(episodes_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch_idx + 1, "task_type": args.task, **result}) + "\n")

            if idx % 25 == 0 or idx == len(sample_ids):
                logger.info(
                    "Epoch %s: %s/%s done, success_rate=%.4f",
                    epoch_idx + 1, idx, len(sample_ids), pass_count / idx,
                )

        logger.info(
            "Epoch %s done: pass=%s/%s (%.4f)",
            epoch_idx + 1,
            pass_count,
            len(sample_ids),
            (pass_count / len(sample_ids)) if sample_ids else 0.0,
        )

        if args.smoke:
            break


if __name__ == "__main__":
    main()
