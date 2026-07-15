import sys
from pathlib import Path
import logging
import argparse
import math
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.service.memory_service import MemoryService
from memrl.agent.bcb_agent import BCBAgent, BCB_SYSTEM_PROMPT
from memrl.episode.agent_runner import EpisodeRunner
from memrl.envs.bcb_episode_adapter import BCBEpisodeEnvAdapter

DEFAULT_SPLIT_FILES = {
    "hard": project_root / "configs" / "bigcodebench" / "splits" / "hard_seed42.json",
    "full": project_root / "configs" / "bigcodebench" / "splits" / "full_seed123.json",
}


def setup_logging(project_root: Path, name: str):
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Logging configured. Log file: {log_filepath}")
    return log_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BigCodeBench (BCB) with the agentic two-tier EpisodeRunner")
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_bcb_config.local.yaml")
            if (project_root / "configs" / "rl_bcb_config.local.yaml").exists()
            else (project_root / "configs" / "rl_bcb_config.yaml")
        ),
    )
    p.add_argument("--subset", type=str, default="full", choices=["hard", "full"])
    p.add_argument("--split", type=str, default="instruct", choices=["instruct", "complete"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_file", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--bcb_repo", type=str, default=str(project_root / "3rdparty" / "bigcodebench-main"))
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--eval_timeout", type=float, default=60.0)
    p.add_argument("--untrusted_hard_timeout", type=float, default=120.0)
    p.add_argument(
        "--init-only",
        action="store_true",
        help="Build the EpisodeRunner and adapter and exit without running episodes.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single batch as a wiring smoke test, then exit.",
    )
    return p.parse_args()


logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    try:
        cfg = MempConfig.from_yaml(args.config)
        setup_logging(project_root, cfg.experiment.experiment_name)

        if args.split_file is None:
            default_split = DEFAULT_SPLIT_FILES.get(args.subset)
            if default_split is not None and default_split.exists():
                args.split_file = str(default_split)

        out_dir = Path(cfg.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime('%Y%m%d-%H%M%S')
        run_dir = out_dir / "bigcodebench" / f"exp_{cfg.experiment.experiment_name}_{run_id}"
        log_dir = run_dir / "local_cache"
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)

        llm_provider = OpenAILLM(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            default_temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
            default_max_tokens=(args.max_tokens if args.max_tokens is not None else cfg.llm.max_tokens),
            token_log_dir=str(log_dir),
        )
        embedding_provider = OpenAIEmbedder(
            api_key=cfg.embedding.api_key,
            base_url=cfg.embedding.base_url,
            model=cfg.embedding.model,
            max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
            token_log_dir=str(log_dir),
        )

        # Same reuse_skill_db/skill_db_path/fresh-per-run pattern as run_alfworld.py.
        if getattr(cfg.memory, "reuse_skill_db", False) and cfg.memory.skill_db_path:
            db_path = Path(cfg.memory.skill_db_path)
            if not db_path.is_absolute():
                db_path = (project_root / db_path).resolve()
        else:
            db_path = run_dir / "skill_memory.sqlite"
        memory_service = MemoryService(
            memory_config=cfg.memory,
            embedding_provider=embedding_provider,
            db_path=str(db_path),
        )

        agent = BCBAgent(
            llm_provider=llm_provider,
            few_shot_examples=[],
            system_prompt=BCB_SYSTEM_PROMPT,
        )

        env_adapter = BCBEpisodeEnvAdapter(
            subset=args.subset,
            split=args.split,
            data_path=args.data_path,
            split_file=args.split_file,
            train_ratio=args.train_ratio,
            seed=args.seed,
            batch_size=int(cfg.experiment.batch_size),
            bcb_repo=args.bcb_repo,
            eval_timeout_s=args.eval_timeout,
            untrusted_hard_timeout_s=args.untrusted_hard_timeout,
        )
        logger.info(
            "Building BCB runner with subset=%s split=%s skill_db=%s output_dir=%s",
            args.subset,
            args.split,
            memory_service.db_path,
            run_dir,
        )

        runner = EpisodeRunner(
            agent=agent,
            memory_service=memory_service,
            sleep_checkpoint=None,  # built from llm_provider inside EpisodeRunner
            env_adapter=env_adapter,
            config=str(args.config),
            output_dir=out_dir,
            experiment_name=cfg.experiment.experiment_name,
            mode=cfg.experiment.mode,
            run_id=run_id,
            run_dir=run_dir,
            retrieve_k=int(cfg.memory.k_retrieve),
            batch_size=int(cfg.experiment.batch_size),
            max_steps=int(cfg.experiment.max_steps),
            llm_provider=llm_provider,
            tensorboard_log_dir=str(tb_dir),
        )
        logger.info("TensorBoard logs will be saved to %s", tb_dir)

        if args.init_only:
            logger.info("EpisodeRunner + BCBEpisodeEnvAdapter initialized; exiting due to --init-only.")
            return

        try:
            num_tasks = env_adapter.num_tasks()
            num_sections = int(args.epochs) * math.ceil(num_tasks / int(cfg.experiment.batch_size)) if num_tasks else 1
            if args.smoke:
                num_sections = 1
            logger.info(
                "Running %s section(s) over %s BCB train tasks (epochs=%s, batch_size=%s).",
                num_sections, num_tasks, args.epochs, cfg.experiment.batch_size,
            )
            for section_idx in range(num_sections):
                summary = runner.run()
                logger.info(
                    "Section %s done: mean_reward=%.4f success_rate=%.4f steps=%.1f "
                    "formation=%s pruning=%s sleep=%s",
                    section_idx + 1,
                    float(summary.get("mean_reward", 0.0)),
                    float(summary.get("success_rate", 0.0)),
                    float(summary.get("mean_steps", 0.0)),
                    summary.get("formation"),
                    summary.get("pruning"),
                    summary.get("sleep_consolidation"),
                )
                if args.smoke:
                    break
        finally:
            runner.close()

    except Exception as e:
        logger.error(f"An unhandled error occurred during the experiment: {e}", exc_info=True)


if __name__ == "__main__":
    main()
