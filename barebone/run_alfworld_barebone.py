"""Barebone ALFWorld runner: plain ReAct, no memory, no tools.

Standalone by design -- does not import EpisodeRunner, MemoryService, or
anything under memrl.agent/memrl.skills. This is the true "No-Memory"
baseline (matching the ReAct-only rows in the papers we've been comparing
against), not a config toggle on top of the memory system.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.envs.alfworld_env import AlfWorldEnv
from memrl.envs.alfworld_episode_adapter import (
    _extract_gamefile,
    _task_description_from_observation,
    _task_type_from_gamefile,
)
from memrl.providers.llm import OpenAILLM

from agent import BarebonAgent

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
    p = argparse.ArgumentParser(description="Barebone ALFWorld ReAct runner -- no memory, no tools.")
    p.add_argument(
        "--alfworld_config",
        type=str,
        default=str(project_root / "configs" / "envs" / "alfworld.yaml"),
    )
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
    )
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument(
        "--num_sections",
        type=int,
        default=10,
        help="Fallback sections-per-epoch if env.num_games can't be resolved. "
        "Normally auto-computed as ceil(num_games / batch_size) so every task "
        "in the split runs exactly once per epoch, no more.",
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--api_key", type=str, default=None, help="Falls back to OPENAI_API_KEY env var.")
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--output_dir", type=str, default=str(project_root / "results" / "barebone"))
    p.add_argument("--smoke", action="store_true", help="Run a single section, then exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("alfworld_barebone")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key: pass --api_key or set OPENAI_API_KEY.")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"exp_{args.mode}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_provider = OpenAILLM(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        default_temperature=args.temperature,
        default_max_tokens=args.max_tokens,
        token_log_dir=str(run_dir),
    )

    env = AlfWorldEnv(config_path=args.alfworld_config, task_type=args.mode, batch_size=args.batch_size)
    agents = [BarebonAgent(llm_provider) for _ in range(args.batch_size)]

    episodes_path = run_dir / "episodes.jsonl"

    # Sections-per-epoch is the minimum number of full batch_size sections
    # needed to touch every game in the split at least once. When num_games
    # isn't evenly divisible by batch_size, the last section of an epoch
    # necessarily wraps around and re-dispatches a few already-seen games
    # to fill the batch -- seen_gamefiles (reset each epoch, below) tracks
    # this so those repeats are excluded from reported metrics and from
    # episodes.jsonl instead of being double-counted.
    num_games = env.num_games
    if args.smoke:
        num_epochs, sections_per_epoch = 1, 1
    elif num_games:
        num_epochs = args.num_epochs
        sections_per_epoch = math.ceil(num_games / args.batch_size)
        logger.info(
            "num_games=%s batch_size=%s -> %s section(s)/epoch",
            num_games, args.batch_size, sections_per_epoch,
        )
    else:
        logger.warning("env.num_games returned nothing; falling back to --num_sections=%s per epoch", args.num_sections)
        num_epochs, sections_per_epoch = args.num_epochs, args.num_sections

    try:
        section_counter = 0
        for epoch_idx in range(num_epochs):
            seen_gamefiles: set = set()
            for _ in range(sections_per_epoch):
                section_counter += 1
                reset_results = env.reset()

                batch_size = len(reset_results)
                observations = [r["obs"] for r in reset_results]
                done_flags = [False] * batch_size
                rewards = [0.0] * batch_size
                step_counts = [0] * batch_size
                task_types = [None] * batch_size
                task_descriptions = [None] * batch_size
                duplicate_flags = [False] * batch_size
                admissible_commands_list = [[] for _ in range(batch_size)]

                for i, entry in enumerate(reset_results):
                    info = entry.get("info") or {}
                    gamefile = _extract_gamefile(info)
                    task_types[i] = _task_type_from_gamefile(gamefile)
                    task_descriptions[i] = _task_description_from_observation(entry.get("obs"))
                    admissible_commands_list[i] = list(info.get("admissible_commands") or [])
                    is_duplicate = gamefile is not None and gamefile in seen_gamefiles
                    if gamefile is not None and not is_duplicate:
                        seen_gamefiles.add(gamefile)
                    duplicate_flags[i] = is_duplicate

                for i, agent in enumerate(agents):
                    agent.reset(task_descriptions[i] or "")

                for _ in range(args.max_steps):
                    if all(done_flags):
                        break

                    actions = []
                    for i in range(batch_size):
                        if done_flags[i]:
                            actions.append("look")
                            continue
                        actions.append(agents[i].act(observations[i], admissible_commands_list[i]))

                    step_results = env.step(actions)
                    for i, result in enumerate(step_results):
                        if done_flags[i]:
                            continue
                        observations[i] = result["obs"]
                        rewards[i] += float(result.get("reward", 0.0) or 0.0)
                        done_flags[i] = bool(result.get("done", False))
                        step_counts[i] += 1
                        admissible_commands_list[i] = list((result.get("info") or {}).get("admissible_commands") or [])

                counted_slots = [i for i in range(batch_size) if not duplicate_flags[i]]
                success_count = sum(1 for i in counted_slots if done_flags[i] and rewards[i] > 0)
                mean_reward = sum(rewards[i] for i in counted_slots) / len(counted_slots) if counted_slots else 0.0
                mean_steps = sum(step_counts[i] for i in counted_slots) / len(counted_slots) if counted_slots else 0.0
                success_rate = success_count / len(counted_slots) if counted_slots else 0.0

                logger.info(
                    "Epoch %s section %s done: mean_reward=%.4f success_rate=%.4f mean_steps=%.1f duplicate_slots=%s",
                    epoch_idx + 1, section_counter, mean_reward, success_rate, mean_steps,
                    batch_size - len(counted_slots),
                )

                with open(episodes_path, "a", encoding="utf-8") as f:
                    for i in counted_slots:
                        f.write(json.dumps({
                            "epoch": epoch_idx + 1,
                            "section": section_counter,
                            "slot": i,
                            "task_type": task_types[i],
                            "task_name": task_descriptions[i],
                            "reward": rewards[i],
                            "success": bool(done_flags[i] and rewards[i] > 0),
                            "steps": step_counts[i],
                        }) + "\n")

                if args.smoke:
                    break
            if args.smoke:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
