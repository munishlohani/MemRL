from __future__ import annotations

import argparse
import copy
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.agent.memp_agent import MempAgent
from memrl.configs.config import MempConfig
from memrl.episode.agent_runner import EpisodeRunner
from memrl.envs.alfworld_episode_adapter import AlfWorldEpisodeEnvAdapter
from memrl.providers.embedding import OpenAIEmbedder
from memrl.providers.llm import OpenAILLM
from memrl.service.memory_service import MemoryService


def setup_logging(project_root: Path, name: str) -> None:
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info("Logging configured. Log file: %s", log_filepath)


def _default_base_config_path(project_root: Path) -> Path:
    local_path = project_root / "configs" / "rl_alf_config.local.yaml"
    if local_path.exists():
        return local_path
    return project_root / "configs" / "rl_alf_config.yaml"


def _default_ray_config_path(project_root: Path) -> Path:
    local_path = project_root / "configs" / "rl_alf_ray.local.yml"
    if local_path.exists():
        return local_path
    return project_root / "configs" / "rl_alf_ray.yml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_trials(ray_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    trials = ray_cfg.get("trials")
    if isinstance(trials, list) and trials:
        normalized: List[Dict[str, Any]] = []
        for idx, trial in enumerate(trials):
            if not isinstance(trial, dict):
                continue
            normalized.append(
                {
                    "name": str(trial.get("name") or f"trial-{idx}"),
                    "overrides": dict(trial.get("overrides") or {}),
                }
            )
        if normalized:
            return normalized

    return [
        {
            "name": str(ray_cfg.get("name") or "alfworld-ray"),
            "overrides": dict(ray_cfg.get("overrides") or {}),
        }
    ]


def _write_resolved_config(
    *,
    base_config_path: Path,
    overrides: Dict[str, Any],
    work_dir: Path,
) -> Path:
    merged = _merge_dicts(_load_yaml(base_config_path), overrides)
    resolved_path = work_dir / "resolved_config.yaml"
    with open(resolved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return resolved_path


def _build_runner(cfg: MempConfig, *, config_path: Path, run_root: Path) -> EpisodeRunner:
    log_dir = run_root / "local_cache"
    log_dir.mkdir(parents=True, exist_ok=True)

    llm_provider = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=cfg.llm.temperature,
        default_max_tokens=cfg.llm.max_tokens,
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
        fallback = project_root / "configs" / "alfworld" / "alfworld_examples.json"
        if fallback.exists():
            logging.getLogger(__name__).warning(
                "few_shot_path %s missing; using %s", few_shot_path, fallback
            )
            few_shot_path = fallback
    if few_shot_path.exists():
        with open(few_shot_path, "r", encoding="utf-8") as f:
            few_shot_examples = json.load(f)
    else:
        few_shot_examples = []

    agent = MempAgent(llm_provider=llm_provider, few_shot_examples=few_shot_examples)
    env_adapter = AlfWorldEpisodeEnvAdapter(
        config_path=str(project_root / "configs" / "envs" / "alfworld.yaml"),
        task_type=str(cfg.experiment.mode or "train"),
        batch_size=int(cfg.experiment.batch_size),
    )

    return EpisodeRunner(
        agent=agent,
        memory_service=memory_service,
        sleep_checkpoint=None,
        env_adapter=env_adapter,
        config=str(config_path),
        output_dir=run_root.parent.parent,
        experiment_name=cfg.experiment.experiment_name,
        mode=cfg.experiment.mode,
        retrieve_k=int(cfg.memory.k_retrieve),
        batch_size=int(cfg.experiment.batch_size),
        max_steps=int(cfg.experiment.max_steps),
        llm_provider=llm_provider,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ALFWorld with EpisodeRunner from YAML hyperparameters on Ray"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_default_base_config_path(project_root)),
        help="Base MemRL YAML config for the run.",
    )
    parser.add_argument(
        "--ray-config",
        type=str,
        default=str(_default_ray_config_path(project_root)),
        help="YAML file containing Ray execution settings and hyperparameter overrides.",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        help="Optional trial name to run from the ray-config trials list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.config)
    ray_config_path = Path(args.ray_config)
    if not base_config_path.is_absolute():
        base_config_path = (project_root / base_config_path).resolve()
    if not ray_config_path.is_absolute():
        ray_config_path = (project_root / ray_config_path).resolve()

    ray_cfg = _load_yaml(ray_config_path)
    base_cfg_data = _load_yaml(base_config_path)
    base_cfg = MempConfig(**base_cfg_data)
    setup_logging(project_root, base_cfg.experiment.experiment_name)

    trials = _resolve_trials(ray_cfg)
    if args.trial is not None:
        trials = [trial for trial in trials if trial["name"] == args.trial]
        if not trials:
            raise ValueError(f"Trial '{args.trial}' was not found in {ray_config_path}")

    ray_settings = dict(ray_cfg.get("ray") or {})
    ray_address = ray_settings.get("address")
    local_mode = bool(ray_settings.get("local_mode", False))
    ignore_reinit_error = bool(ray_settings.get("ignore_reinit_error", True))

    try:
        import ray
    except Exception as exc:
        raise RuntimeError("Ray is required to use run/run_alfworld_ray.py") from exc

    ray.init(
        address=ray_address or None,
        local_mode=local_mode,
        ignore_reinit_error=ignore_reinit_error,
        log_to_driver=bool(ray_settings.get("log_to_driver", True)),
    )

    resources = dict(ray_settings.get("resources_per_trial") or {})
    if not resources:
        resources = {"num_cpus": 1}

    remote_worker = ray.remote(**resources)(_run_trial)
    object_refs = []
    for trial in trials:
        object_refs.append(
            remote_worker.remote(
                trial_name=str(trial["name"]),
                base_config_path=str(base_config_path),
                overrides=trial["overrides"],
            )
        )

    results = ray.get(object_refs)
    for result in results:
        logging.getLogger(__name__).info(
            "Trial %s finished: mean_reward=%.4f success_rate=%.4f steps=%.1f",
            result.get("trial_name"),
            float(result.get("mean_reward", 0.0)),
            float(result.get("success_rate", 0.0)),
            float(result.get("mean_steps", 0.0)),
        )


def _run_trial(
    *,
    trial_name: str,
    base_config_path: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"memp_alfworld_{trial_name}_") as tmpdir:
        tmp_path = Path(tmpdir)
        resolved_config_path = _write_resolved_config(
            base_config_path=Path(base_config_path),
            overrides=overrides,
            work_dir=tmp_path,
        )
        cfg = MempConfig.from_yaml(str(resolved_config_path))
        setup_logging(project_root, f"alfworld_{trial_name}")
        run_root = Path(cfg.experiment.output_dir) / "alfworld" / trial_name
        run_root.mkdir(parents=True, exist_ok=True)
        runner = _build_runner(cfg, config_path=resolved_config_path, run_root=run_root)
        summary = runner.run()
        return {
            "trial_name": trial_name,
            "mean_reward": summary.get("mean_reward", 0.0),
            "mean_steps": summary.get("mean_steps", 0.0),
            "success_rate": summary.get("success_rate", 0.0),
        }


if __name__ == "__main__":
    main()
