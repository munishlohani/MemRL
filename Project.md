# Agent Memory with Utility-Based Skill Consolidation
## Architecture Specification — Phase 1
**Working Paper | Summer 2026**

---

## Abstract

We propose a memory architecture for AI agents that organizes skills within a single unified hierarchical graph, where structural depth encodes estimated transferability. Skills are formed from experience through a multi-stage gating pipeline that separates novelty detection, utility verification, statistical confidence, and stability estimation before any transferability claim is made. Utility is estimated online via Q-learning with temporal difference updates per task type, and memory retention follows a biologically-grounded Ebbinghaus decay formula modulated by **global weighted-mean utility** — not task-local utility — consistent with the unified (non-partitioned) graph design. The agent's memory management uses two complementary mechanisms: a continuous decay-based pruning threshold and a hard action space bound. The full system is framed as an extended MDP over states, actions, transitions, rewards, a discount factor, and an external memory bank.

**Base template:** MemRL. This architecture extends MemRL by replacing flat memory with a hierarchical skill graph, introducing a gated formation pipeline, and separating decay salience (global) from utility estimation (per-task-type).

> **Status:** Formalism complete. Node schema finalized for Phase 1. Retrieval technique is the one remaining open design decision. Ready for implementation.

---

## Implementation Notes for Coding Agent

This section provides a compact map of every component so the coding agent can implement without ambiguity. Read this before touching any other section.

**What the system is:** A skill memory graph sitting alongside an LLM-based agent. The agent solves tasks by selecting skills from the graph (via retrieval), executing them, receiving rewards, and using those rewards to update Q-values. Separately, the graph grows (skill formation), shrinks (decay + pruning), and reorganizes (float-up/demotion) based on accumulated statistics.

**Key data structures:**
- `SkillNode` — the primary data object. One per skill. Defined in §6.3.
- `CandidateRecord` — a lightweight pre-graph accumulator. Lives in `candidate_pool`. Defined in §4.2.
- `SkillGraph` — a tree over `SkillNode` objects. Owns the `children_index`. Defined in §6.1.
- `EpisodicMemoryBank` — separate store of raw experiences. Linked from nodes via `evidence_ids`. Not part of the graph.

**Execution order per episode:**
1. Classify task type → `t_k`
2. For each step: retrieve candidates → select skill → execute → compute TD error → update Q → run Gate 1
3. End of episode: run Gate 2 (candidate → node) → run graph maintenance (decay + pruning + float-up) → recompute `decay_rate` for all active nodes

**What MemRL provides vs. what this adds:**
- MemRL provides: basic memory bank, retrieval by similarity, episode-level updates
- This adds: hierarchical graph structure, gated formation pipeline, transferability scoring, depth-indexed decay with global salience denominator, bidirectional float-up/demotion with hysteresis

**Critical invariants the coding agent must preserve:**
- `decay_rate` on a node always equals `λ_d / (Q̄_w + ε)`. Recompute after every Q-update to an active node.
- `children_index` lives on the graph object, not the node. Node stores only `parent_id`.
- `total_accessed` = `sum(self.n.values())`. Expose as a `@property`, never store separately.
- All new nodes enter at `depth = 3`. No exceptions.
- The pruning loop uses `node.decay_rate` directly — no `t_k` dependency in pruning.
- `d = 1` nodes are never pruned regardless of retention value.

---

## 1. Introduction and Motivation

Standard agent memory systems conflate several distinct questions:

- Which experience is worth storing?
- Which stored experience is worth keeping?
- Which kept experience generalizes to new tasks?

Most prior work (MemGPT, A-MEM, Voyager, SkillLib) optimizes retrieval — choosing what to surface at inference time — but treats memory formation and consolidation as secondary. This work inverts the priority: the primary contribution is a principled answer to **which experiences should consolidate into reusable skills, and at what level of generality**.

The central hypothesis is that not all useful skills are transferable. A skill may exhibit high utility within a single task distribution while remaining highly task-specific. Organizing memory by estimated transferability — rather than recency, retrieval frequency, or semantic similarity — produces a graph whose structure reflects genuine generalizability rather than usage patterns.

**Key design decisions confirmed for Phase 1:**

- A single unified hierarchical graph. Depth encodes transferability continuously; there is no partition into separate graph objects per task type.
- Skills are formed through a two-gate pipeline (Gate 1: TD error novelty → Gate 2: utility pre-filter). Transferability gating (Gates 3–4) governs float-up, not formation.
- Utility is estimated via Q-learning with TD updates, stored **per task type**. This per-type granularity is required for transferability scoring.
- Memory decay uses **global weighted-mean utility** $\bar{Q}_{i,w}$ in the denominator — not task-specific $Q_i(t_k)$ — because the graph is unified and decay governs global graph membership, not per-task relevance.
- Two memory management mechanisms coexist: a decay-based pruning threshold $\theta_{\text{prune}}$ (soft) and a hard action space cap $|A| \leq N$ (hard).

**Explicitly deferred to Phase 2:**

- Affect/personalization graph
- DAG extension for multi-parent nodes
- Memory-quality bonus term in reward
- Double Q-learning for overestimation bias correction

---

## 2. Problem Formulation

### 2.1 MDP Definition

$$\mathcal{MDP} = \left(S,\ A,\ P,\ R,\ \gamma,\ \mathcal{M}\right)$$

The memory bank $\mathcal{M}$ is a **side-channel** that conditions the policy. It is not part of the state space $S$ — it has independent update dynamics. Embedding $\mathcal{M}$ in $S$ would make the state space grow with every new skill, creating a non-stationary MDP with no convergence guarantees.

### 2.2 State

$$s_t = \left(t_k,\ c_t,\ h_t\right)$$

| Component | Description |
|---|---|
| $t_k$ | Task type. Fixed within an episode. Changes between episodes. |
| $c_t$ | Task context at step $t$ — the current problem being solved. |
| $h_t$ | Short-term interaction history over the last $w$ steps. |

Task type $t_k$ is the primary conditioning variable for utility estimation and retrieval. Its formal definition — benchmark-derived, cluster-assigned, or hierarchical taxonomy — is an open problem noted in §11.

### 2.3 Action

$$a_t = s_i \in \mathcal{M}$$

The agent selects which skill to retrieve and apply. Token-level generation is handled by the underlying LLM and is outside this MDP. The skill-level MDP operates at reasoning-step granularity.

### 2.4 Transition

$$s_{t+1} = \left(t_k,\ c_{t+1},\ h_{t+1}\right)$$

$c_{t+1}$ reflects the outcome of applying $s_i$ to the current context. History updates as $h_{t+1} = h_t \cup \{(s_i, r_t)\}$. Task type $t_k$ is invariant within an episode.

### 2.5 Reward

$$r_t = r_t^{\text{env}}$$

Environment feedback per reasoning step. A memory-quality bonus is defined but deferred:

$$r_t^{\text{full}} = r_t^{\text{env}} + \beta \cdot r_t^{\text{mem}}, \qquad \beta = 0 \text{ in Phase 1}$$

### 2.6 Discount Factor

$$\gamma \in [0.9,\ 0.99]$$

High because a skill deployed now may enable better subsequent skills.

### 2.7 Memory Bank

$$\mathcal{M}_t = \left(\mathcal{G}_t,\ \{Q_i(t_k)\},\ \{\lambda_d\},\ \epsilon\right)$$

| Component | Description |
|---|---|
| $\mathcal{G}_t$ | Unified skill graph at time $t$ |
| $\{Q_i(t_k)\}$ | All Q-estimates, indexed by skill and task type |
| $\{\lambda_d\}$ | Depth-indexed base decay rates (not baking in Q) |
| $\epsilon$ | Utility floor for decay denominator |

$\mathcal{M}$ is updated **after each episode**, not per step.

---

## 3. Utility Estimation

### 3.1 Semantics

$$Q_i(t_k)\ \approx\ \mathbb{E}\!\left[\Delta R \mid s_i,\ t_k\right], \qquad \Delta R = R_{\text{with skill}} - R_{\text{without skill}}$$

Q-values are stored **per task type**, never as a global scalar. Per-type granularity is required for the transferability estimator.

### 3.2 Q-learning Update Rule

$$Q_i(t_k)\ \leftarrow\ Q_i(t_k)\ +\ \alpha \Bigl[r_t\ +\ \gamma \max_{s_j \in \mathcal{N}(s_i)} Q_j(t_k)\ -\ Q_i(t_k)\Bigr]$$

| Term | Description |
|---|---|
| $\alpha$ | Learning rate. Starting value: $0.1$. |
| $r_t$ | Per-step environment reward. |
| $\gamma$ | Discount factor. |
| $\mathcal{N}(s_i)$ | Local neighborhood: parent node + child nodes of $s_i$. |

This is Q-learning (off-policy): the update uses the greedy policy over the local neighborhood regardless of which skill was actually selected.

After a Q-update to node $s_i$, immediately call `s_i.recompute_decay_rate()` (see §7.4). This keeps `decay_rate` fresh at the cost of one arithmetic operation per active node per episode.

### 3.3 TD Error

$$\delta_t = r_t + \gamma \max_{s_j \in \mathcal{N}(s_i)} Q_j(t_k) - Q_i(t_k)$$

$\delta_t$ drives the Q-update and serves as the novelty signal for Gate 1.

### 3.4 Initialization

$$Q_i(t_k) = 0 \quad \forall\, t_k$$

No utility prior at creation.

### 3.5 Action Selection

$$a_t = \arg\max_{s_i \in \text{top-}k}\ Q_i(t_k)$$

Candidates are shortlisted by semantic similarity before Q-ranking. Retrieval technique is an open decision (§11).

### 3.6 Failure Credit Assignment

When a negative surprise ($\delta_t < -\theta_\delta$) occurs at step $t$, the penalty is distributed across all skills active in the episode using **recency-weighted credit**:

$$\Delta Q_{s}(t_k) = -|\delta_t| \cdot \gamma^{T - \text{step}(s)}$$

where $T$ is the current step and $\text{step}(s)$ is the step at which skill $s$ was last active. Skills used most recently before the failure receive the largest penalty; earlier skills receive exponentially smaller corrections. This prevents good setup-skills from being uniformly penalized for a failure they did not cause.

> **Implementation note:** Maintain `active_skills: list[tuple[SkillNode, int]]` — tuples of (node, step_index) — during each episode. On negative Gate 1 trigger, iterate this list and apply the weighted penalty.

---

## 4. Gated Skill Formation Pipeline

Raw experiences do not directly become skill nodes. A two-gate formation pipeline controls admission and node creation. Transferability gates (3–4) govern float-up separately and are documented in §6.4.

```
Raw experience
      ↓
  Gate 1: TD error (novelty + sign check)
      ↓ (positive surprise only)
  Candidate pool (accumulation)
      ↓
  Gate 2: Utility pre-filter
      ↓
  Skill node created at d=3
      ↓
  [Gates 3–4 govern float-up, not formation — see §6.4]
```

### 4.1 Gate 1 — Experience Admission (TD Error)

$$|\delta_t| > \theta_\delta$$

**Sign semantics:**

| Condition | Interpretation | Action |
|---|---|---|
| $\delta_t > \theta_\delta$ | Positive surprise — better than expected | Add to candidate pool |
| $\delta_t < -\theta_\delta$ | Negative surprise — worse than expected | Apply recency-weighted penalty to active skills (§3.6) |
| $|\delta_t| \leq \theta_\delta$ | Expected outcome | Discard entirely |

Failure experiences do **not** form candidates. Their only role is Q-correction via §3.6.

### 4.2 Candidate Accumulation

Admitted experiences accumulate in a pre-graph `candidate_pool`. Each `CandidateRecord` tracks:

- `n_i` — number of successful activations across episodes
- `Q_mean_i` — running **unweighted** mean Q-estimate (cheap; no shrinkage at this stage)
- `task_type` — task type at first admission (for provenance)
- `last_seen_episode` — for pool-level decay (candidates that go cold are evicted)

> **Implementation note:** `CandidateRecord` is a lightweight dataclass, not a `SkillNode`. It has no embedding and no graph position. It gets garbage-collected if `last_seen_episode` exceeds a staleness threshold without passing Gate 2.

### 4.3 Gate 2 — Utility Pre-Filter (Node Creation)

$$n_i > N_{\text{skill}} \quad \text{AND} \quad \bar{Q}_i > \theta_U$$

- $N_{\text{skill}}$ — minimum activation count (separate from $N_{\min}$ used in float-up)
- $\theta_U$ — minimum mean utility threshold
- Unweighted mean used here — cheap pre-filter; Bayesian shrinkage is deferred to transferability scoring

Upon passing, a `SkillNode` is instantiated at `depth = 3` and inserted into the graph. The `CandidateRecord` is removed from the pool.

> **Implementation note:** Node creation requires LLM skill extraction to populate `content` and an embedding model call to populate `embedding`. Both are I/O operations — batch them at end-of-episode, not inline during the step loop.

---

## 5. Transferability — Full Formalism

Transferability scoring governs float-up only. It is never used for pruning or formation decisions.

### 5.1 Confidence-Weighted Variance

$$\text{Var}_w(Q_i) = \sum_{k=1}^{K} w_{ik} \cdot \left(Q_i(t_k) - \bar{Q}_{i,w}\right)^2$$

### 5.2 Bayesian Shrinkage Weights

$$w_{ik} = \frac{n_{ik}}{n_{ik} + \lambda}, \qquad \lambda = 10$$

| Limit | Behavior |
|---|---|
| $n_{ik} \to \infty$ | $w_{ik} \to 1$ — fully trust the estimate |
| $n_{ik} \to 0$ | $w_{ik} \to 0$ — distrust sparse observations |

Grounded in Empirical Bayes shrinkage (Efron & Morris, 1977).

### 5.3 Weighted Mean

$$\bar{Q}_{i,w} = \frac{\sum_{k=1}^{K} w_{ik} \cdot Q_i(t_k)}{\sum_{k=1}^{K} w_{ik}}$$

**Cold-start behavior:** For a node observed only on task type $t_{k_0}$, the shrinkage weights cancel in the ratio and $\bar{Q}_{i,w} = Q_i(t_{k_0})$. The formula is well-defined from the first retrieval.

### 5.4 Transferability Score

$$\hat{T}(s_i) = \frac{\bar{Q}_{i,w}^2}{\bar{Q}_{i,w}^2 + \text{Var}_w(Q_i)}$$

Only computed after Gates 3 and 4 pass (see §6.4). A signal-to-noise ratio: high-utility and consistent = high $\hat{T}$. High-utility but inconsistent = low $\hat{T}$.

| $\hat{T}(s_i)$ | Interpretation |
|---|---|
| $\to 1.0$ | Consistent utility across task types — highly generalizable |
| $\approx 0.5$ | Moderate cross-task variance — semi-generalizable |
| $\to 0.0$ | High variance — task-specific |

---

## 6. Unified Skill Graph

### 6.1 Structure

$$\mathcal{G} = (V,\ E)$$

| Component | Description |
|---|---|
| $V$ | All `SkillNode` objects plus one virtual root $r$ |
| $E$ | Parent → child directed edges |
| Parent constraint | Each node has exactly **one parent** (tree, not DAG, in Phase 1) |
| Cross-edges | None in Phase 1 |

`children_index: dict[str, set[str]]` is owned by the **graph object**, not by individual nodes. Nodes store only `parent_id`. This prevents bidirectional pointer inconsistency during reparenting — the graph object performs atomic updates to `children_index` and `node.parent_id` together.

**Phase 2 extension:** `secondary_parents: list[str]` is reserved on each node (empty in Phase 1) for DAG promotion at $d = 2$.

### 6.2 Depth Assignment

$$\text{depth}(s_i) = \begin{cases} 0 & \text{virtual root — structural anchor only} \\ 1 & \hat{T}(s_i) \geq \theta_1 \quad \text{highly general, permanent} \\ 2 & \theta_2 \leq \hat{T}(s_i) < \theta_1 \quad \text{semi-general} \\ 3 & \hat{T}(s_i) < \theta_2 \quad \text{task-specific (leaf)} \end{cases}$$

**Starting thresholds:** $\theta_1 = 0.75$, $\theta_2 = 0.40$. Both swept in ablation.

**Invariant:** All new nodes enter at $d = 3$. Depth can only decrease (float up) as evidence accumulates.

### 6.3 Node Schema — FINALIZED for Phase 1

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SkillNode:
    # --- Identity ---
    id: str                          # UUID, assigned at creation

    # --- Skill Representation ---
    content: str                     # LLM-generated procedural summary of the skill
    embedding: np.ndarray            # Dense vector; used for retrieval ranking and
                                     # parent-finding on float-up

    # --- Provenance ---
    task_type_primary: str           # Task type t_k under which skill was first formed
    t_create: int                    # Global retrieval step at creation

    # --- Hierarchy ---
    depth: int                       # Current depth ∈ {1, 2, 3}. Always 3 at creation.
    parent_id: str | None            # UUID of parent node. None only for virtual root.
    secondary_parents: list[str] = field(default_factory=list)
                                     # Reserved for Phase 2 DAG extension. Empty in Phase 1.

    # --- Usage Statistics ---
    last_accessed_step: int = 0      # Global step index of most recent retrieval.
                                     # Used to compute Δt in decay formula.

    # --- Utility Tracking ---
    Q: dict[str, float] = field(default_factory=dict)
                                     # Q(t_k): per-task-type Q-values.
                                     # Keys are task type strings.
    n: dict[str, int] = field(default_factory=dict)
                                     # n(t_k): retrieval counts per task type.
                                     # Used for shrinkage weights and Gate 3.

    # --- Retention ---
    decay_rate: float = 0.0          # Cached value of λ_d / (Q̄_w + ε).
                                     # GLOBAL salience denominator — uses weighted-mean
                                     # utility across all task types, NOT task-specific Q.
                                     # Recomputed after every Q-update via recompute_decay_rate().
                                     # At d=1, always 0.0 (no decay).

    # --- Episodic Links ---
    evidence_ids: list[str] = field(default_factory=list)
                                     # IDs into the EpisodicMemoryBank.
                                     # Capped at R entries via reservoir sampling.
                                     # Provides diagnostic trace back to raw experiences.

    # --- Derived Properties ---
    @property
    def total_accessed(self) -> int:
        """Total retrievals across all task types. Derived — never stored separately."""
        return sum(self.n.values())

    def recompute_decay_rate(self, lambda_d: float, epsilon: float) -> None:
        """
        Recompute and cache the global decay rate.

        decay_rate = λ_d / (Q̄_w + ε)

        where Q̄_w is the confidence-weighted mean utility across ALL task types.
        This is the GLOBAL salience denominator — consistent with the unified
        (non-partitioned) graph. Task-specific Q-values drive utility estimation
        and transferability scoring; the global mean drives retention.

        Call this after every Q-update to any task type on this node.
        At d=1, sets decay_rate = 0.0 unconditionally.
        """
        if self.depth == 1:
            self.decay_rate = 0.0
            return
        Q_bar_w = self._weighted_mean_utility(lambda_shrink=10)
        self.decay_rate = lambda_d / (Q_bar_w + epsilon)

    def _weighted_mean_utility(self, lambda_shrink: float = 10) -> float:
        """Bayesian shrinkage weighted mean: Q̄_w = Σ w_ik Q(t_k) / Σ w_ik."""
        if not self.Q:
            return 0.0
        weighted_sum = 0.0
        weight_sum = 0.0
        for t_k, q in self.Q.items():
            n_ik = self.n.get(t_k, 0)
            w = n_ik / (n_ik + lambda_shrink)
            weighted_sum += w * q
            weight_sum += w
        if weight_sum == 0.0:
            return 0.0
        return weighted_sum / weight_sum
```

**Field-by-field notes for the coding agent:**

| Field | Notes |
|---|---|
| `content` | Open: raw reasoning trace vs. distilled procedural summary. Distilled summary preferred for retrieval quality. LLM extraction prompt TBD. |
| `embedding` | Open: frozen LLM encoder vs. fine-tuned. Populated at node creation. Shape depends on encoder choice. |
| `decay_rate` | Always equals `λ_d / (Q̄_w + ε)`. Stale by at most one episode. Never compute retention inline without calling `recompute_decay_rate()` first. |
| `evidence_ids` | Reservoir-sampled. Implement `add_evidence(eid)` with reservoir sampling at cap $R$. $R$ is a hyperparameter (suggested starting: 50). |
| `secondary_parents` | Do not read or write in Phase 1. Initialize empty. |
| `total_accessed` | A `@property`. Do not add a stored counter — it will diverge. |

### 6.4 Float-Up Mechanism (Gates 3 and 4)

```python
def maybe_float_up(node: SkillNode, graph: SkillGraph, K: int,
                   N_min: int, theta_CV: float,
                   theta_1: float, theta_2: float,
                   epsilon_hyst: float) -> None:

    # Gate 3: Confidence — sufficient cross-task evidence
    if node.total_accessed < N_min:          # N_min = 5K
        return

    # Gate 4: Stability — CV of utility across task types
    cv = compute_cv(node)                    # sqrt(Var_w) / Q̄_w
    if cv >= theta_CV:
        return

    # Compute transferability score
    T_hat = compute_transferability(node)    # Q̄_w² / (Q̄_w² + Var_w)

    target_depth = depth_from_T(T_hat, theta_1, theta_2)

    if target_depth < node.depth:
        # Reparent: find highest cosine-similarity node at (target_depth - 1)
        new_parent = graph.find_best_parent(node, target_depth - 1)
        graph.reparent(node, new_parent)     # atomic: updates children_index + parent_id
        node.depth = target_depth
        node.recompute_decay_rate(graph.lambda_d[node.depth], graph.epsilon)
```

`find_best_parent` uses cosine similarity over `embedding`. This is the only structural use of semantic similarity — retrieval uses it for ranking; float-up uses it for graph rewiring.

### 6.5 Demotion and Hysteresis

$$\text{demote}(s_i) \iff \hat{T}(s_i) < \theta_{\text{lower}}(d) - \epsilon_{\text{hyst}}$$

where $\theta_{\text{lower}}(d)$ is the lower boundary of the current depth band and $\epsilon_{\text{hyst}} = 0.1$.

A recently promoted node is exempt from demotion for $M_{\text{wait}}$ episodes (grace period, prevents oscillation).

---

## 7. Memory Decay

### 7.1 Design Rationale: Global vs. Task-Local Decay

The graph is **unified** — there is no separate graph per task type. Decay governs global graph membership: whether a node survives at all. It is therefore architectural incoherent to decay a node using task-local $Q_i(t_k)$, which would make the same node's retention path-dependent on whatever task type happened to run last.

The correct salience denominator is the **global weighted-mean utility** $\bar{Q}_{i,w}$ — a task-agnostic measure of how much the skill contributes across all contexts the agent has encountered. This is the same quantity used in transferability scoring, so no new computation is required.

### 7.2 Formula

$$d_i(\Delta t) = e^{-\lambda_d \cdot \Delta t\, /\, (\bar{Q}_{i,w} + \epsilon)}$$

| Term | Description |
|---|---|
| $\lambda_d$ | Depth-indexed base decay rate (§7.3) |
| $\Delta t$ | Elapsed retrieval steps since `last_accessed_step` (not wall-clock time) |
| $\bar{Q}_{i,w}$ | Confidence-weighted mean utility across all task types (§5.3) |
| $\epsilon$ | Floor term preventing division by zero. Starting value: $0.01$ |

**Time reference:** $\Delta t$ is in retrieval steps, not wall-clock time. Wall-clock penalizes idle periods where the agent had no retrieval opportunity.

**Decay rate:** The effective decay rate for node $s_i$ is $\lambda_d / (\bar{Q}_{i,w} + \epsilon)$. This is cached as `node.decay_rate`. Retention at query time is then simply:

$$d_i(\Delta t) = e^{-\text{decay\_rate} \cdot \Delta t}$$

### 7.3 Depth-Indexed Base Decay Rate

$$\lambda_d = \begin{cases} 0 & d = 1 \quad \text{permanent — no decay} \\ \lambda_{\text{slow}} & d = 2 \\ \lambda_{\text{fast}} & d = 3 \end{cases}$$

$$\lambda_{\text{fast}} = 5 \cdot \lambda_{\text{slow}}$$

$d = 1$ nodes have `decay_rate = 0.0` unconditionally. They are never pruned.

### 7.4 `recompute_decay_rate` Protocol

After any Q-update to a node that was active in an episode:

```python
node.recompute_decay_rate(lambda_d=graph.lambda_d[node.depth], epsilon=graph.epsilon)
```

After a float-up (depth changes, so $\lambda_d$ changes):

```python
node.depth = target_depth
node.recompute_decay_rate(lambda_d=graph.lambda_d[node.depth], epsilon=graph.epsilon)
```

`recompute_decay_rate` is called on **active nodes only** (those retrieved this episode), not on the entire graph. This is $O(\text{active})$, not $O(|G|)$.

### 7.5 Boundary Cases

| Condition | Effective rate | Consequence |
|---|---|---|
| $\bar{Q}_{i,w} = 0$, $d = 3$ | $\lambda_{\text{fast}} / \epsilon$ — very fast | New skills with no confirmed utility are quickly pruned |
| $\bar{Q}_{i,w} \to 1$, $d = 3$ | $\approx \lambda_{\text{fast}}$ | High-utility task-specific skills decay at base rate |
| $d = 1$ | $0$ | Permanent |
| Single task type observed | $\bar{Q}_{i,w} = Q_i(t_{k_0})$ (shrinkage cancels) | Cold-start is well-defined — collapses to home task type |

---

## 8. Memory Management

Two complementary mechanisms control active skill set size.

### 8.1 Mechanism 1 — Decay-Based Pruning

$$d_i(\Delta t) < \theta_{\text{prune}}$$

A node is removed when its retention falls below $\theta_{\text{prune}}$. This is **task-agnostic**: uses `node.decay_rate`, not a task-specific Q-value. The pruning loop has no dependency on the current episode's $t_k$.

```python
# Graph maintenance loop — task-agnostic
for node in graph.nodes:
    if node.depth == 1:
        continue                          # permanent; never pruned
    delta_t = current_step - node.last_accessed_step
    retention = exp(-node.decay_rate * delta_t)
    if retention < theta_prune:
        graph.remove(node)
        continue
    maybe_float_up(node, ...)
```

Note: no `t_k` appears in this loop. This is correct and intentional.

### 8.2 Mechanism 2 — Action Space Cap

$$|A| \leq N$$

At retrieval time, only the top-$N$ nodes per depth by current Q-value are eligible. This is a hard computational guarantee on action space size, required for Q-learning convergence (fixed finite action space assumption).

**Interaction with decay:** Decay governs whether a node *exists* in the graph. The $N$-cap governs whether an existing node is *reachable* by the retrieval policy. A node can exist but be excluded from the action space if it ranks outside top-$N$ at its depth.

| Mechanism | Type | Controls | Hyperparameter |
|---|---|---|---|
| Ebbinghaus decay + $\theta_{\text{prune}}$ | Soft, continuous | Graph membership | $\theta_{\text{prune}}$ |
| $|A| \leq N$ | Hard, discrete | Retrieval eligibility | $N$ |

---

## 9. Retrieval

> ⚠️ **OPEN — NOT FINALIZED.** Confirmed constraints below. Exact method (ANN vs. BM25 vs. hybrid) is the one remaining open design decision.

### 9.1 Confirmed Constraints

1. **Bottom-up traversal.** Start at $d = 3$ leaf nodes matching $t_k$, walk ancestor chains toward $d = 1$.

2. **Retrieval score formula:**

$$\text{score}(s_i,\ \Delta t) = d_i(\Delta t) \cdot \cos(e_i,\ e_q)$$

   where $d_i(\Delta t) = e^{-\text{decay\_rate} \cdot \Delta t}$ (uses cached `decay_rate`) and $\cos(e_i, e_q)$ is cosine similarity between skill embedding and query embedding.

3. **Action space bound respected.** Only top-$N$ nodes per depth eligible.

4. **Traversal cost:** $O(m \cdot D)$ where $m$ = shortlist size, $D = 3$.

### 9.2 Conceptual Direction

Leading candidate: **bottom-up semantic search with ancestor expansion.**

- Step 1: Semantic shortlist of $m$ leaf nodes at $d = 3$ tagged with $t_k$
- Step 2: Collect ancestor chains of all shortlisted nodes
- Step 3: Score all candidates by retrieval score formula
- Step 4: Return top-$k$

---

## 10. Episode Update Loop

```python
# G               — SkillGraph object
# candidate_pool  — dict[str, CandidateRecord]
# current_step    — global retrieval step counter
# active_skills   — list of (SkillNode, step_index) tuples for credit assignment

for each episode:
    t_k = classify_task(episode)
    active_skills = []

    for each step t in episode:

        # RETRIEVAL
        candidates = recall(query=c_t, task_type=t_k, k=5)
        a_t = argmax_{s_i in candidates} Q_i[t_k]
        active_skills.append((a_t, t))

        # EXECUTION
        r_t, s_{t+1} = env.step(a_t)

        # TD UPDATE
        delta_t = r_t + gamma * max_{s_j in N(a_t)} Q_j[t_k] - Q_{a_t}[t_k]
        Q_{a_t}[t_k] += alpha * delta_t
        a_t.n[t_k] = a_t.n.get(t_k, 0) + 1
        a_t.last_accessed_step = current_step
        a_t.recompute_decay_rate(lambda_d[a_t.depth], epsilon)  # update global salience cache

        # GATE 1 — Experience Admission
        if delta_t > theta_delta:
            candidate_pool.add_or_update(experience(s_t, a_t, r_t, t_k))
        elif delta_t < -theta_delta:
            # Recency-weighted failure credit (§3.6)
            T = current_step
            for (s, step_s) in active_skills:
                penalty = abs(delta_t) * (gamma ** (T - step_s))
                Q_s[t_k] -= penalty
                s.recompute_decay_rate(lambda_d[s.depth], epsilon)

        current_step += 1

    # END OF EPISODE

    # Gate 2 — Candidate → Node (batch LLM extraction here)
    for m_i in list(candidate_pool.values()):
        if m_i.n > N_skill and m_i.Q_mean > theta_U:
            new_node = create_skill_node(m_i)          # LLM summary + embedding call
            parent = G.find_best_parent(new_node, depth=2)
            G.insert(new_node, parent)
            del candidate_pool[m_i.id]

    # Graph maintenance — task-agnostic, no t_k dependency
    for node in list(G.nodes):
        if node.depth == 1:
            continue
        delta_t_node = current_step - node.last_accessed_step
        retention = exp(-node.decay_rate * delta_t_node)
        if retention < theta_prune:
            G.remove(node)
            continue
        maybe_float_up(node, G, K, N_min, theta_CV, theta_1, theta_2, epsilon_hyst)
```

---

## 11. Open Problems

| Item | Status | Notes |
|---|---|---|
| Retrieval technique | **Open** | ANN (FAISS/ScaNN) vs. BM25 vs. hybrid; shortlist method TBD |
| Content representation | **Open** | Raw trace vs. distilled procedural summary vs. structured template |
| Embedding strategy | **Open** | Frozen LLM encoder vs. fine-tuned vs. task-conditioned |
| Task type definition $t_k$ | **Open** | Benchmark-derived, clustered, or hierarchical taxonomy |
| Skill extraction method | **Open** | LLM prompt + output format for procedural skill summary |
| Evidence reservoir size $R$ | **Open** | Suggested starting: 50 per node |
| DAG extension | **Deferred Phase 2** | Multi-parent nodes at $d = 2$ |
| Affect/personalization graph | **Deferred Phase 2** | Volatile user-preference memory |
| Double Q-learning | **Deferred Phase 2** | Overestimation bias correction |
| Memory-quality reward bonus | **Deferred Phase 2** | $r_t^{\text{mem}} = Q_i(t_k) - \bar{Q}(t_k)$ |

---

## 12. Hyperparameter Summary

| Symbol | Role | Starting value | Status |
|---|---|---|---|
| $\theta_\delta$ | TD error admission threshold | — | sweep |
| $N_{\text{skill}}$ | Min activations for node creation (Gate 2) | — | sweep |
| $\theta_U$ | Min mean utility for node creation (Gate 2) | — | sweep |
| $N_{\min}$ | Min cross-task evidence for float-up (Gate 3) | $5K$ | derived from $K$ |
| $\theta_{\text{CV}}$ | Max coefficient of variation (Gate 4) | — | sweep |
| $\theta_1$ | Transferability cutoff for $d = 1$ | $0.75$ | sweep |
| $\theta_2$ | Transferability cutoff for $d = 2$ | $0.40$ | sweep |
| $\lambda$ | Bayesian shrinkage pseudocount | $10$ | sweep |
| $\lambda_{\text{slow}}$ | Base decay rate at $d = 2$ | — | sweep |
| $\lambda_{\text{fast}}$ | Base decay rate at $d = 3$ | $5 \times \lambda_{\text{slow}}$ | derived |
| $\epsilon$ | Utility floor in decay denominator | $0.01$ | sweep |
| $\theta_{\text{prune}}$ | Retention threshold for node removal | — | sweep |
| $N$ | Hard action space cap per depth | — | sweep |
| $\alpha$ | TD learning rate | $0.1$ | sweep |
| $\gamma$ | Discount factor | $[0.9, 0.99]$ | sweep |
| $\epsilon_{\text{hyst}}$ | Demotion hysteresis buffer | $0.1$ | sweep |
| $M_{\text{wait}}$ | Grace period episodes after promotion | — | sweep |
| $R$ | Evidence reservoir size per node | $50$ | sweep |

$K$ in $N_{\min} = 5K$ is not a hyperparameter — it is the observed count of distinct task types and grows during training.

---

## 13. Relationship to MemRL

| Aspect | MemRL | This Work |
|---|---|---|
| Memory structure | Flat bank | Hierarchical skill graph (depth encodes transferability) |
| Skill formation | All experiences stored | Two-gate pipeline (novelty + utility pre-filter) |
| Retention | Recency / retrieval frequency | Ebbinghaus decay modulated by global weighted-mean utility |
| Generalization | None (flat retrieval) | Transferability scoring via confidence-weighted variance (Gates 3–4) |
| Graph structure | None | Unified tree; float-up/demotion with hysteresis |
| Action space | Unbounded | Hard cap $|A| \leq N$ per depth + soft decay pruning |
| Utility signal | Global scalar | Per-task-type Q-values via TD learning |
| Decay salience | N/A | $\bar{Q}_{i,w}$ — global, task-agnostic; consistent with unified graph |