"""Barebone AppWorld runner: plain multi-turn code-execution agent, no memory, no tools.

Standalone by design -- does not import EpisodeRunner, MemoryService, or
anything under memrl.agent/memrl.skills/memrl.envs.appworld_episode_adapter.
This is the true "No-Memory" baseline.

Reuses AppWorldClient (the subprocess protocol client, not the memory-aware
adapter) because that subprocess boundary exists for pydantic/SQLAlchemy
version isolation between MemRL's venv and `.venv-appworld`, not for
memory -- the barebone ALFWorld runner reuses AlfWorldEnv for the identical
reason (env plumbing is not part of what "no memory" strips out).

Sequential, single worker: the memory-aware adapter keeps one AppWorld
worker process per parallel batch slot. This baseline drives one worker and
one task at a time instead, matching run_bcb_barebone.py's plain sequential
loop rather than adding parallel-worker bookkeeping a baseline doesn't need.

Per-episode loop mirrors AppWorldEpisodeEnvAdapter.step(): repeatedly send
the agent's code to the worker until it calls apis.supervisor.complete_task
(task_completed=True) or the step budget runs out, then score once via a
single evaluate() call -- AppWorld's TestTracker only reports a verdict for
the world as it now stands, so scoring mid-episode would be meaningless.
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

from memrl.appworld_eval.client import AppWorldClient, AppWorldClientError
from memrl.providers.llm import OpenAILLM

from agent import BarebonAppWorldAgent

logger = logging.getLogger(__name__)

_VALID_SPLITS = ("train", "dev", "test_normal", "test_challenge")


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
    p = argparse.ArgumentParser(description="Barebone AppWorld runner -- no memory, one worker, one task at a time.")
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=list(_VALID_SPLITS),
        help="AppWorld split (train=90, dev=57, test_normal=168, test_challenge=417 tasks).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--appworld_python",
        type=str,
        default=str(project_root / ".venv-appworld" / "bin" / "python"),
        help="Interpreter with `appworld` installed -- must NOT be MemRL's own venv.",
    )
    p.add_argument(
        "--appworld_root",
        type=str,
        default=str(project_root / "data" / "appworld"),
        help="Data root from `appworld download data --root <root>`.",
    )
    p.add_argument("--max_steps", type=int, default=30, help="Turn budget per episode. Matches rl_appworld_config.yaml.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--api_key", type=str, default=None, help="Falls back to OPENAI_API_KEY env var.")
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--max_tokens",
        type=int,
        default=10240,
        help="Matches rl_appworld_config.yaml's llm.max_tokens -- code+reasoning turns need more room than ALFWorld/BCB.",
    )
    p.add_argument("--output_dir", type=str, default=str(project_root / "results" / "barebone"))
    p.add_argument("--smoke", action="store_true", help="Run a single task, then exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("appworld_barebone")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key: pass --api_key or set OPENAI_API_KEY.")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"exp_appworld_{args.split}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_provider = OpenAILLM(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        default_temperature=args.temperature,
        default_max_tokens=args.max_tokens,
        token_log_dir=str(run_dir),
    )
    agent = BarebonAppWorldAgent(llm_provider)

    client = AppWorldClient(
        python_executable=args.appworld_python,
        appworld_root=args.appworld_root,
        experiment_name="barebone",
    )

    episodes_path = run_dir / "episodes.jsonl"

    try:
        try:
            task_ids = client.task_ids(args.split)
        except AppWorldClientError as exc:
            raise SystemExit(str(exc))

        if not task_ids:
            raise SystemExit(f"No tasks found for split={args.split!r}.")

        num_epochs = 1 if args.smoke else int(args.epochs)
        run_task_ids = task_ids[:1] if args.smoke else task_ids
        logger.info(
            "Running %s epoch(s) over %s AppWorld %s task(s) (model=%s).",
            num_epochs, len(run_task_ids), args.split, args.model,
        )

        for epoch_idx in range(num_epochs):
            pass_count = 0
            for idx, task_id in enumerate(run_task_ids, start=1):
                try:
                    reset_info = client.reset(task_id)
                except AppWorldClientError as exc:
                    logger.error("AppWorld reset failed for task_id=%s: %s", task_id, exc)
                    with open(episodes_path, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "epoch": epoch_idx + 1,
                                    "task_id": task_id,
                                    "status": "RESET_ERROR",
                                    "error": str(exc),
                                    "steps": 0,
                                    "task_completed": False,
                                    "reward": 0.0,
                                    "success": False,
                                }
                            )
                            + "\n"
                        )
                    continue

                agent.reset(str(reset_info.get("instruction") or ""))

                observation = ""
                steps = 0
                completed = False
                status = "STEP_BUDGET_EXHAUSTED"
                error = None

                while steps < args.max_steps:
                    code = agent.act(observation)
                    try:
                        result = client.execute(code)
                    except AppWorldClientError as exc:
                        logger.error("AppWorld execute failed for task_id=%s: %s", task_id, exc)
                        status, error = "EXECUTE_ERROR", str(exc)
                        break
                    steps += 1
                    observation = str(result.get("output") or "")
                    completed = bool(result.get("task_completed"))
                    if completed:
                        status = "COMPLETED"
                        break

                success = False
                reward = 0.0
                eval_info: dict = {}
                if error is None:
                    try:
                        evaluation = client.evaluate()
                        success = bool(evaluation.get("success"))
                        reward = 1.0 if success else 0.0
                        eval_info = {
                            "pass_count": evaluation.get("pass_count"),
                            "fail_count": evaluation.get("fail_count"),
                            "num_tests": evaluation.get("num_tests"),
                            "pass_percentage": evaluation.get("pass_percentage"),
                            "difficulty": evaluation.get("difficulty"),
                        }
                    except AppWorldClientError as exc:
                        logger.error("AppWorld evaluate failed for task_id=%s: %s", task_id, exc)
                        error = f"evaluate_error: {exc}"

                client.close_world()
                pass_count += 1 if success else 0

                with open(episodes_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "epoch": epoch_idx + 1,
                                "task_id": task_id,
                                "status": status,
                                "error": error,
                                "steps": steps,
                                "task_completed": completed,
                                "reward": reward,
                                "success": success,
                                **eval_info,
                            }
                        )
                        + "\n"
                    )

                if idx % 10 == 0 or idx == len(run_task_ids):
                    logger.info(
                        "Epoch %s: %s/%s done, pass@1=%.4f",
                        epoch_idx + 1, idx, len(run_task_ids), pass_count / idx,
                    )

            logger.info(
                "Epoch %s done: pass=%s/%s (%.4f)",
                epoch_idx + 1,
                pass_count,
                len(run_task_ids),
                (pass_count / len(run_task_ids)) if run_task_ids else 0.0,
            )

            if args.smoke:
                break
    finally:
        client.close()


if __name__ == "__main__":
    main()
