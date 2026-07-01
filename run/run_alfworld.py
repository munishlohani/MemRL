import sys
from pathlib import Path
import logging
import argparse
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.agent.prompts import ALFWORLD_SYSTEM_PROMPT
from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.service.memory_service import MemoryService
from memrl.agent.memp_agent import MempAgent
from memrl.episode.agent_runner import EpisodeRunner
from memrl.envs.alfworld_episode_adapter import AlfWorldEpisodeEnvAdapter


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


def _resolve_alfworld_config_path(cfg: MempConfig) -> Path:
    path = Path(cfg.environment.alfworld_config_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AlfWorld benchmark with the agentic two-tier EpisodeRunner")
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_alf_config.local.yaml")
            if (project_root / "configs" / "rl_alf_config.local.yaml").exists()
            else (project_root / "configs" / "rl_alf_config.yaml")
        ),
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument(
        "--init-only",
        action="store_true",
        help="Build the EpisodeRunner and adapter and exit without running episodes.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single episode as a wiring smoke test, then exit.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to run in the outer loop. Defaults to experiment.num_sections.",
    )
    return p.parse_args()


logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    try:
        cfg = MempConfig.from_yaml(args.config)
        setup_logging(project_root, cfg.experiment.experiment_name)

        out_dir = Path(cfg.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime('%Y%m%d-%H%M%S')
        run_dir = out_dir / "alfworld" / f"exp_{cfg.experiment.experiment_name}_{run_id}"
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

        memory_service = MemoryService(
            memory_config=cfg.memory,
            embedding_provider=embedding_provider,
            db_path=cfg.memory.skill_db_path,
        )

        import json
        few_shot_path = project_root / cfg.experiment.few_shot_path
        if not few_shot_path.exists():
            # Config default points at data/alfworld/, but the file ships under
            # configs/alfworld/. Fall back so wiring is testable without the
            # exact data layout.
            fallback = project_root / "configs" / "alfworld" / "alfworld_examples.json"
            if fallback.exists():
                logger.warning("few_shot_path %s missing; using %s", few_shot_path, fallback)
                few_shot_path = fallback
        if few_shot_path.exists():
            with open(few_shot_path, "r", encoding="utf-8") as f:
                few_shot_examples = json.load(f)
        else:
            logger.warning("Few-shot examples not found at %s; proceeding with empty examples.", few_shot_path)
            few_shot_examples = []
        agent = MempAgent(
            llm_provider=llm_provider,
            few_shot_examples=few_shot_examples,
            system_prompt=ALFWORLD_SYSTEM_PROMPT,
        )

        alfworld_config_path = _resolve_alfworld_config_path(cfg)
        env_adapter = AlfWorldEpisodeEnvAdapter(
            config_path=str(alfworld_config_path),
            task_type=str(cfg.experiment.mode or "train"),
            batch_size=int(cfg.experiment.batch_size),
        )
        logger.info(
            "Building ALFWorld runner with env_config=%s skill_db=%s output_dir=%s",
            alfworld_config_path,
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
            retrieve_k=int(cfg.memory.k_retrieve),
            batch_size=int(cfg.experiment.batch_size),
            max_steps=int(cfg.experiment.max_steps),
            llm_provider=llm_provider,
            tensorboard_log_dir=str(tb_dir),
        )
        logger.info("TensorBoard logs will be saved to %s", tb_dir)

        if args.init_only:
            logger.info("EpisodeRunner + AlfWorldEpisodeEnvAdapter initialized; exiting due to --init-only.")
            return

        num_episodes = 1 if args.smoke else (args.episodes or cfg.experiment.num_sections)
        logger.info("Running %s episode(s) on ALFWorld via EpisodeRunner.", num_episodes)
        for episode_idx in range(int(num_episodes)):
            summary = runner.run()
            logger.info(
                "Episode %s done: mean_reward=%.4f success_rate=%.4f steps=%.1f "
                "formation=%s pruning=%s sleep=%s",
                episode_idx + 1,
                float(summary.get("mean_reward", 0.0)),
                float(summary.get("success_rate", 0.0)),
                float(summary.get("mean_steps", 0.0)),
                summary.get("formation"),
                summary.get("pruning"),
                summary.get("sleep_consolidation"),
            )
            if args.smoke:
                break

    except Exception as e:
        logger.error(f"An unhandled error occurred during the experiment: {e}", exc_info=True)


if __name__ == "__main__":
    main()
