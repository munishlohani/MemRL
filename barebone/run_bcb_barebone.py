"""Barebone BigCodeBench runner: one LLM call per task, no memory, no tools.

Standalone by design -- does not import EpisodeRunner, MemoryService, or
anything under memrl.agent/memrl.skills/memrl.envs.bcb_episode_adapter. This
is the true "No-Memory" baseline. Unlike the ALFWorld barebone runner, there
is no ReAct-style step loop here: BigCodeBench is itself a single-step task
(submit code, evaluate, done), so "barebone" for BCB is simply one prompt in,
one code submission out, per task -- not an iterative loop with memory
stripped out.
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

from memrl.bigcodebench_eval.eval_utils import (
    ensure_bigcodebench_on_path,
    run_untrusted_check_with_hard_timeout,
    sanitize_code,
)
from memrl.bigcodebench_eval.task_wrappers import get_prompt, load_bcb_data, split_dataset
from memrl.providers.llm import OpenAILLM

from agent import BarebonBCBAgent

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_FILES = {
    "hard": project_root / "configs" / "bigcodebench" / "splits" / "hard_seed42.json",
    "full": project_root / "configs" / "bigcodebench" / "splits" / "full_seed123.json",
}


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
    p = argparse.ArgumentParser(description="Barebone BigCodeBench runner -- no memory, one LLM call per task.")
    p.add_argument("--subset", type=str, default="full", choices=["hard", "full"])
    p.add_argument("--split", type=str, default="instruct", choices=["instruct", "complete"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Path to a JSON split file with train_ids/val_ids. Defaults to the same "
        "legacy split files under configs/bigcodebench/splits/ that run_bcb.py uses.",
    )
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--bcb_repo", type=str, default=str(project_root / "3rdparty" / "bigcodebench-main"))
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--api_key", type=str, default=None, help="Falls back to OPENAI_API_KEY env var.")
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1280)
    p.add_argument("--eval_timeout", type=float, default=60.0)
    p.add_argument("--untrusted_hard_timeout", type=float, default=120.0)
    p.add_argument("--output_dir", type=str, default=str(project_root / "results" / "barebone"))
    p.add_argument("--smoke", action="store_true", help="Run a single task, then exit.")
    return p.parse_args()


def evaluate_solution(
    *,
    task: dict,
    code: str,
    bcb_repo: str,
    eval_timeout_s: float,
    untrusted_hard_timeout_s: float,
) -> dict:
    """Compile-check -> sanitize -> untrusted_check -> status mapping.

    Same sequence as BCBEpisodeEnvAdapter._evaluate / the original
    BCBRunner._evaluate_one -- duplicated here (not imported) to keep this
    script standalone, matching how the two memory-aware copies are
    themselves independent of each other.
    """
    task_id = str(task.get("task_id", "unknown"))
    entry_point = str(task.get("entry_point", "task_func"))
    test_code = str(task.get("test", "") or "")

    if not test_code:
        return {"task_id": task_id, "status": "SYNTAX_OK", "error": "no_test_code"}

    try:
        compile(code, "<string>", "exec")
    except SyntaxError as exc:
        return {"task_id": task_id, "status": "SYNTAX_ERROR", "error": str(exc)}

    clean_code = sanitize_code(code, entry_point, bcb_repo=bcb_repo)

    from bigcodebench.eval import FAIL, PASS, TIMEOUT  # type: ignore

    stat, details, err, hard_timed_out = run_untrusted_check_with_hard_timeout(
        code=clean_code,
        test_code=test_code,
        entry_point=entry_point,
        max_as_limit=30 * 1024,
        max_data_limit=30 * 1024,
        max_stack_limit=10,
        min_time_limit=1.0,
        gt_time_limit=eval_timeout_s,
        hard_timeout_s=untrusted_hard_timeout_s,
        bcb_repo=bcb_repo,
    )

    if hard_timed_out:
        return {"task_id": task_id, "status": "TIMEOUT", "error": err or "hard_timeout"}
    if err:
        return {"task_id": task_id, "status": "RUNTIME_ERROR", "error": err}
    if stat == PASS:
        return {"task_id": task_id, "status": "PASS"}
    if stat == TIMEOUT:
        return {"task_id": task_id, "status": "TIMEOUT", "error": "timeout"}
    if stat == FAIL:
        return {"task_id": task_id, "status": "FAIL", "error": str(details)[:500] if details else "fail"}
    return {"task_id": task_id, "status": "UNKNOWN", "error": str(stat)}


def main() -> None:
    args = parse_args()
    setup_logging("bcb_barebone")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key: pass --api_key or set OPENAI_API_KEY.")

    if args.split_file is None:
        default_split = DEFAULT_SPLIT_FILES.get(args.subset)
        if default_split is not None and default_split.exists():
            args.split_file = str(default_split)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"exp_bcb_{args.subset}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_provider = OpenAILLM(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        default_temperature=args.temperature,
        default_max_tokens=args.max_tokens,
        token_log_dir=str(run_dir),
    )
    agent = BarebonBCBAgent(llm_provider)

    ensure_bigcodebench_on_path(args.bcb_repo)
    problems = load_bcb_data(subset=args.subset, data_path=args.data_path)
    train_ids, _val_ids, _test_ids = split_dataset(
        problems,
        train_ratio=args.train_ratio,
        seed=args.seed,
        split_file=args.split_file,
    )

    num_epochs = 1 if args.smoke else int(args.epochs)
    task_ids = train_ids[:1] if args.smoke else train_ids
    logger.info(
        "Running %s epoch(s) over %s BCB train task(s) (subset=%s, split=%s, model=%s).",
        num_epochs, len(task_ids), args.subset, args.split, args.model,
    )

    episodes_path = run_dir / "episodes.jsonl"

    for epoch_idx in range(num_epochs):
        pass_count = 0
        for idx, task_id in enumerate(task_ids, start=1):
            task = problems[task_id]
            prompt = get_prompt(task, split=args.split)

            code = agent.act(prompt)
            eval_result = evaluate_solution(
                task=task,
                code=code,
                bcb_repo=args.bcb_repo,
                eval_timeout_s=args.eval_timeout,
                untrusted_hard_timeout_s=args.untrusted_hard_timeout,
            )
            ok = eval_result.get("status") == "PASS"
            pass_count += 1 if ok else 0

            with open(episodes_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch_idx + 1,
                            "task_id": task_id,
                            "status": eval_result.get("status"),
                            "error": eval_result.get("error"),
                            "pass": ok,
                        }
                    )
                    + "\n"
                )

            if idx % 25 == 0 or idx == len(task_ids):
                logger.info(
                    "Epoch %s: %s/%s done, pass@1=%.4f",
                    epoch_idx + 1, idx, len(task_ids), pass_count / idx,
                )

        logger.info(
            "Epoch %s done: pass=%s/%s (%.4f)",
            epoch_idx + 1,
            pass_count,
            len(task_ids),
            (pass_count / len(task_ids)) if task_ids else 0.0,
        )

        if args.smoke:
            break


if __name__ == "__main__":
    main()
