"""Shared utilities for Q-value estimation and manipulation.

This module owns the shrinkage-weighted estimators used by the memory graph,
retrieval, and consolidation code paths.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple


def compute_shrinkage_weighted_mean_from_samples(
    samples: Sequence[Tuple[float, int]],
    lambda_shrink: float = 10.0,
) -> float:
    """Compute a shrinkage-weighted mean from explicit value/count samples."""
    if not samples:
        return 0.0

    weighted_sum = 0.0
    weight_sum = 0.0
    for value, count in samples:
        count_f = float(count)
        weight = count_f / (count_f + lambda_shrink)
        weighted_sum += weight * float(value)
        weight_sum += weight

    if weight_sum == 0.0:
        return 0.0

    return weighted_sum / weight_sum


def compute_shrinkage_weighted_mean(
    value_dict: Dict[str, float],
    count_dict: Dict[str, int],
    lambda_shrink: float = 10.0,
) -> float:
    """
    Compute shrinkage-weighted mean across task types.

    This is the canonical implementation used for both tactical Q-values
    and strategic Q_omega values. The shrinkage formula is:

        w_tk = n_tk / (n_tk + lambda_shrink)
        mean = sum(w_tk * value[tk]) / sum(w_tk)

    The lambda_shrink parameter acts as a pseudocount, dampening the weight
    of rarely-observed task types. When a task type is observed once (n_tk=1),
    its weight is 1/(1+lambda_shrink) ≈ 0.09 (for lambda_shrink=10).

    Args:
        value_dict: Dict mapping task_type → value (Q or Q_omega).
        count_dict: Dict mapping task_type → count (n or n_omega).
        lambda_shrink: Shrinkage pseudocount (default 10.0).

    Returns:
        Shrinkage-weighted mean, or 0.0 if value_dict is empty.

    Examples:
        >>> # Tactical Q-values across task types
        >>> q_dict = {"pick_and_place": 0.8, "look_at_obj": 0.5}
        >>> n_dict = {"pick_and_place": 5, "look_at_obj": 1}
        >>> mean_q = compute_shrinkage_weighted_mean(q_dict, n_dict)
        >>> # "pick_and_place" has higher count, so weights it more heavily

        >>> # Strategic Q_omega values
        >>> q_omega_dict = {"pick_and_place": 0.6, "clean": 0.4}
        >>> n_omega_dict = {"pick_and_place": 2, "clean": 0}
        >>> mean_q_omega = compute_shrinkage_weighted_mean(q_omega_dict, n_omega_dict)
    """
    samples = [
        (float(value), int(count_dict.get(task_type, 0) or 0))
        for task_type, value in value_dict.items()
    ]
    return compute_shrinkage_weighted_mean_from_samples(
        samples,
        lambda_shrink=lambda_shrink,
    )


def get_q_salience(node: Any, lambda_shrink: float = 10.0) -> float:
    """
    Get decay salience from a tactical node.

    The decay salience is the shrinkage-weighted mean Q-value across all
    observed task types. High-utility nodes decay slowly; low-utility nodes
    are pruned quickly.

    Args:
        node: SkillNode object with Q and n attributes.
        lambda_shrink: Shrinkage pseudocount.

    Returns:
        Shrinkage-weighted mean Q-value (salience).
    """
    q_values = getattr(node, "Q", None) or {}
    n_values = getattr(node, "n", None) or {}

    return compute_shrinkage_weighted_mean(q_values, n_values, lambda_shrink)


def get_q_omega_salience(node: Any, lambda_shrink: float = 10.0) -> float:
    """
    Get option-value salience from a strategic node.

    Used as a fallback when the current task type is not in Q_omega
    (cold-start task type selection).

    Args:
        node: SkillNode object with Q_omega and n_omega attributes.
        lambda_shrink: Shrinkage pseudocount.

    Returns:
        Shrinkage-weighted mean Q_omega (salience).
    """
    q_omega_values = getattr(node, "Q_omega", None) or {}
    n_omega_values = getattr(node, "n_omega", None) or {}

    return compute_shrinkage_weighted_mean(
        q_omega_values, n_omega_values, lambda_shrink
    )


def get_expected_option_value(
    node: Any,
    task_type: str | None = None,
    normalize_by_total_counts: bool = False,
) -> float:
    """
    Get expected option value from a strategic node.

    Implements two modes:
    1. If task_type is specified and found in Q_omega, return that value directly.
    2. Otherwise, return the mean Q_omega weighted by episode counts per task type.

    Args:
        node: SkillNode object with Q_omega and n_omega attributes.
        task_type: Optional specific task type to look up.
        normalize_by_total_counts: If False (default), use shrinkage weighting.
                                   If True, weight by episode count only.

    Returns:
        Option value (float).

    Examples:
        >>> # Bootstrap (task_type not yet observed)
        >>> q_omega = {"pick_and_place": 0.7}
        >>> n_omega = {"pick_and_place": 1}
        >>> node.Q_omega = q_omega
        >>> node.n_omega = n_omega
        >>> value = get_expected_option_value(node, task_type="look_at_obj")
        >>> # Returns mean over all observed types

        >>> # Cold task type fallback for selection
        >>> value = get_expected_option_value(node)  # Mean across all types
    """
    q_omega = getattr(node, "Q_omega", None) or {}

    # Mode 1: task_type specified and observed
    if task_type is not None and task_type in q_omega:
        return float(q_omega.get(task_type, 0.0) or 0.0)

    # Mode 2: return expected value across all observed types
    if not q_omega:
        return 0.0

    n_omega = getattr(node, "n_omega", None) or {}

    if normalize_by_total_counts:
        # Weight by episode count
        total_counts = float(sum(int(v) for v in n_omega.values()))
        if total_counts <= 0.0:
            return 0.0

        expected = 0.0
        for t_k, q_val in q_omega.items():
            weight = float(n_omega.get(t_k, 0) or 0) / total_counts
            expected += weight * float(q_val)
        return expected
    else:
        # Use shrinkage weighting (more robust for rare task types)
        return compute_shrinkage_weighted_mean(q_omega, n_omega)


__all__ = [
    "compute_shrinkage_weighted_mean_from_samples",
    "compute_shrinkage_weighted_mean",
    "get_q_salience",
    "get_q_omega_salience",
    "get_expected_option_value",
]
