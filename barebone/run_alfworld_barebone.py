"""Barebone ALFWorld runner: plain ReAct, no memory, no tools.

Standalone by design -- does not import EpisodeRunner, MemoryService, or
anything under memrl.agent/memrl.skills. This is the true "No-Memory"
baseline (matching the ReAct-only rows in the papers we've been comparing
against), not a config toggle on top of the memory system.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.envs.alfworld_env import AlfWorldEnv
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
    p.add_argument("--num_sections", type=int, default=10)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--api_key", type=str, default=None, help="Falls back to OPENAI_API_KEY env var.")
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--output_dir", type=str, default=str(project_root / "results" / "barebone"))
    p.add_argument("--smoke", action="store_true", help="Run a single section, then exit.")
    return p.parse_args()


def main() -> None:
    import os

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
    num_sections = 1 if args.smoke else args.num_sections

    try:
        for section_idx in range(num_sections):
            reset_results = env.reset()
            for agent in agents:
                agent.reset()

            batch_size = len(reset_results)
            observations = [r["obs"] for r in reset_results]
            done_flags = [False] * batch_size
            rewards = [0.0] * batch_size
            step_counts = [0] * batch_size

            for _ in range(args.max_steps):
                if all(done_flags):
                    break

                actions = []
                for i in range(batch_size):
                    if done_flags[i]:
                        actions.append("look")
                        continue
                    actions.append(agents[i].act(observations[i]))

                step_results = env.step(actions)
                for i, result in enumerate(step_results):
                    if done_flags[i]:
                        continue
                    observations[i] = result["obs"]
                    rewards[i] += float(result.get("reward", 0.0) or 0.0)
                    done_flags[i] = bool(result.get("done", False))
                    step_counts[i] += 1

            success_count = sum(1 for i in range(batch_size) if done_flags[i] and rewards[i] > 0)
            mean_reward = sum(rewards) / batch_size if batch_size else 0.0
            mean_steps = sum(step_counts) / batch_size if batch_size else 0.0
            success_rate = success_count / batch_size if batch_size else 0.0

            logger.info(
                "Section %s done: mean_reward=%.4f success_rate=%.4f mean_steps=%.1f",
                section_idx + 1, mean_reward, success_rate, mean_steps,
            )

            import json
            with open(episodes_path, "a", encoding="utf-8") as f:
                for i in range(batch_size):
                    f.write(json.dumps({
                        "section": section_idx + 1,
                        "slot": i,
                        "reward": rewards[i],
                        "success": bool(done_flags[i] and rewards[i] > 0),
                        "steps": step_counts[i],
                    }) + "\n")

            if args.smoke:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
