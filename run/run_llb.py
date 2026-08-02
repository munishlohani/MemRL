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
from memrl.agent.llb_agent import LLBAgent, LLB_SYSTEM_PROMPT
from memrl.episode.agent_runner import EpisodeRunner
from memrl.envs.llb_episode_adapter import LLBEpisodeEnvAdapter
from memrl.run.checkpoint_utils import load_checkpoint, save_epoch_checkpoint
from memrl.run.training_summary import EpochStatsAccumulator, write_training_summary

LLB_SKILL_CONTRACT_PATH = str(
    project_root / "memrl" / "skills" / "memory_retrieval_skill" / "LLB_SKILL.md"
)


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
    p = argparse.ArgumentParser(description="Run LifelongBench (LLB) with the agentic two-tier EpisodeRunner")
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_llb_config.local.yaml")
            if (project_root / "configs" / "rl_llb_config.local.yaml").exists()
            else (project_root / "configs" / "rl_llb_config.yaml")
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
        help="Run a single section as a wiring smoke test, then exit.",
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
        run_dir = out_dir / "llb" / f"exp_{cfg.experiment.experiment_name}_{run_id}"
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

        # Resuming from a checkpoint always reuses that checkpoint's skill
        # DB (continuing the same graph is the whole point of resuming),
        # taking priority over the reuse_skill_db/fresh-per-run choice below.
        resume_state = None
        if getattr(cfg.experiment, "ckpt_resume_enabled", False) and cfg.experiment.ckpt_resume_path:
            resume_state = load_checkpoint(
                cfg.experiment.ckpt_resume_path, cfg.experiment.ckpt_resume_epoch
            )

        # Same reuse_skill_db/skill_db_path/fresh-per-run pattern as
        # run_alfworld.py/run_bcb.py.
        if resume_state is not None:
            db_path = Path(resume_state["db_path"])
        elif getattr(cfg.memory, "reuse_skill_db", False) and cfg.memory.skill_db_path:
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

        # No few_shot_examples: LLB has no few-shot mechanism (CustomAgent
        # discards that parameter unconditionally, and LLBAgent's own
        # prompting never references it).
        agent = LLBAgent(
            llm_provider=llm_provider,
            system_prompt=LLB_SYSTEM_PROMPT,
        )

        env_adapter = LLBEpisodeEnvAdapter(
            task=cfg.experiment.task,
            train_file=cfg.experiment.split_file,
            val_file=cfg.experiment.valid_file,
            batch_size=int(cfg.experiment.batch_size),
            max_steps=int(cfg.experiment.max_steps),
        )
        logger.info(
            "Building LLB runner with task=%s split_file=%s skill_db=%s output_dir=%s",
            cfg.experiment.task,
            cfg.experiment.split_file,
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
            skill_budget_per_episode=cfg.experiment.skill_budget_per_episode,
            tensorboard_log_dir=str(tb_dir),
            skill_contract_path=LLB_SKILL_CONTRACT_PATH,
        )
        logger.info("TensorBoard logs will be saved to %s", tb_dir)

        resume_epoch_start = 0
        if resume_state is not None:
            runner.load_checkpoint_state(resume_state.get("runner_state") or {})
            resume_epoch_start = int(resume_state.get("epoch", -1)) + 1
            logger.info(
                "Resumed runner state from checkpoint; continuing from epoch %s.",
                resume_epoch_start + 1,
            )

        # Validation: shares agent/memory_service/env_adapter with the train
        # runner (same in-process SkillGraph, so eval sees every node the
        # train runner has formed so far with no reload) but is a distinct
        # EpisodeRunner instance so its own memory_config copy can force
        # build_memory=False -- retrieval/selection still run, but eval
        # never mutates the graph. Skipped for --smoke (fast wiring test),
        # when valid_interval<=0, or when experiment.valid_file is unset
        # (num_val_tasks() returns 0 in that case).
        eval_runner = None
        num_val_sections = 0
        if not args.smoke and cfg.experiment.valid_interval > 0:
            num_val_tasks = env_adapter.num_val_tasks()
            if num_val_tasks <= 0:
                logger.info(
                    "valid_interval=%s but no validation tasks (experiment.valid_file unset "
                    "or empty); periodic validation disabled.",
                    cfg.experiment.valid_interval,
                )
            else:
                eval_tb_dir = tb_dir / "eval"
                eval_tb_dir.mkdir(parents=True, exist_ok=True)
                eval_runner = EpisodeRunner(
                    agent=agent,
                    memory_service=memory_service,
                    sleep_checkpoint=None,  # built from llm_provider inside EpisodeRunner; no-op while build_memory=False
                    env_adapter=env_adapter,
                    config=str(args.config),
                    output_dir=out_dir,
                    experiment_name=f"{cfg.experiment.experiment_name}_eval",
                    mode="eval",
                    run_id=run_id,
                    run_dir=run_dir / "eval",
                    retrieve_k=int(cfg.memory.k_retrieve),
                    batch_size=int(cfg.experiment.batch_size),
                    max_steps=int(cfg.experiment.max_steps),
                    llm_provider=llm_provider,
                    skill_budget_per_episode=cfg.experiment.skill_budget_per_episode,
                    tensorboard_log_dir=str(eval_tb_dir),
                    skill_contract_path=LLB_SKILL_CONTRACT_PATH,
                )
                eval_runner.memory_config.build_memory = False
                num_val_sections = math.ceil(num_val_tasks / int(cfg.experiment.batch_size))
                logger.info(
                    "Validation enabled: %s held-out task(s), %s section(s)/pass, every %s train section(s).",
                    num_val_tasks, num_val_sections, cfg.experiment.valid_interval,
                )

        if args.init_only:
            logger.info("EpisodeRunner + LLBEpisodeEnvAdapter initialized; exiting due to --init-only.")
            return

        def run_validation_pass(after_section: int) -> None:
            env_adapter.set_phase("val")
            env_adapter.reset_epoch_tracking()
            total_reward = 0.0
            total_success = 0.0
            total_episodes = 0
            try:
                for _ in range(num_val_sections):
                    val_summary = eval_runner.run()
                    counted = len(val_summary.get("episodes") or [])
                    total_reward += float(val_summary.get("mean_reward", 0.0)) * counted
                    total_success += float(val_summary.get("success_rate", 0.0)) * counted
                    total_episodes += counted
            finally:
                env_adapter.set_phase("train")
            if total_episodes:
                logger.info(
                    "Validation after train section %s: episodes=%s mean_reward=%.4f success_rate=%.4f",
                    after_section,
                    total_episodes,
                    total_reward / total_episodes,
                    total_success / total_episodes,
                )

        epoch_summaries: list = []
        try:
            num_tasks = env_adapter.num_tasks()
            # "Epochs" for LLB = experiment.num_sections dataset replays
            # (per configs/rl_llb_config.yaml's own documented convention),
            # not experiment.num_epochs like ALFWorld -- LLB's original
            # runner never had a num_epochs field.
            sections_per_epoch = math.ceil(num_tasks / int(cfg.experiment.batch_size)) if num_tasks else 1
            num_epochs = int(cfg.experiment.num_sections)
            if args.smoke:
                num_epochs, sections_per_epoch = 1, 1
            logger.info(
                "Running %s epoch(s) x %s section(s) over %s LLB train tasks (batch_size=%s).",
                num_epochs, sections_per_epoch, num_tasks, cfg.experiment.batch_size,
            )
            if resume_epoch_start >= num_epochs:
                logger.info(
                    "Checkpoint already covers all %s configured epoch(s); nothing to resume.",
                    num_epochs,
                )
            section_counter = 0
            for epoch_idx in range(resume_epoch_start, num_epochs):
                env_adapter.reset_epoch_tracking()
                epoch_stats = EpochStatsAccumulator()
                for _ in range(sections_per_epoch):
                    summary = runner.run()
                    section_counter += 1
                    epoch_stats.add_section(summary)
                    logger.info(
                        "Epoch %s section %s done: mean_reward=%.4f success_rate=%.4f steps=%.1f "
                        "formation=%s pruning=%s sleep=%s",
                        epoch_idx + 1,
                        section_counter,
                        float(summary.get("mean_reward", 0.0)),
                        float(summary.get("success_rate", 0.0)),
                        float(summary.get("mean_steps", 0.0)),
                        summary.get("formation"),
                        summary.get("pruning"),
                        summary.get("sleep_consolidation"),
                    )
                    if args.smoke:
                        break
                    if (
                        eval_runner is not None
                        and cfg.experiment.valid_interval > 0
                        and section_counter % cfg.experiment.valid_interval == 0
                    ):
                        run_validation_pass(section_counter)
                epoch_summaries.append(epoch_stats.finalize(epoch_idx + 1, runner))
                if not args.smoke:
                    save_epoch_checkpoint(run_dir, epoch_idx, runner, db_path)
                if args.smoke:
                    break
        finally:
            runner.close()
            write_training_summary(run_dir, epoch_summaries)

    except Exception as e:
        logger.error(f"An unhandled error occurred during the experiment: {e}", exc_info=True)


if __name__ == "__main__":
    main()
