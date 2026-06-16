# Agent Memory with Utility-Based Skill Consolidation
## Architecture Specification — Phase 1
**Working Paper | Summer 2026**

---

## Abstract

We propose a memory architecture for AI agents that organizes skills within a single unified hierarchical graph, where structural depth encodes estimated transferability and role. Depth 1 ($d=1$) holds **strategic scaffolds** — reasoning frames selected once per episode under an options/semi-MDP formalism — while depths 2–3 hold **tactical skills** — directly executable, per-step actions updated via standard single-step Q-learning. Tactical skills are formed from experience through a two-gate pipeline (novelty, utility pre-filter) and may float up from $d=3$ to $d=2$ via a confidence/stability-gated transferability score. Depth 1 is populated separately, through a periodic **sleep consolidation** event that clusters unabsorbed $d=2$ skills and either merges them into existing scaffolds or spawns new ones — not through continuous per-node float-up. Memory retention follows a biologically-grounded Ebbinghaus decay formula modulated by **global weighted-mean utility**, consistent with the unified (non-partitioned) graph design. The agent's memory management uses two complementary mechanisms: a continuous decay-based pruning threshold and a hard action space bound. The full system is framed as a two-tier extended semi-MDP over states, a partitioned action space (strategic options + tactical actions), transitions, rewards, a discount factor, and an external memory bank.

**Base template:** MemRL. This architecture extends MemRL by replacing flat memory with a hierarchical skill graph, introducing a gated tactical-formation pipeline, a separate sleep-consolidation pipeline for strategic scaffold formation, options-style credit assignment for strategic actions, and separating decay salience (global) from utility estimation (per-task-type).

> **Status:** Formalism complete, including the two-tier action space and sleep-consolidation mechanism. Node schema finalized for Phase 1. Tactical retrieval technique (within $d=2/3$) is the one remaining open design decision.

---

## Implementation Notes for Coding Agent

This section provides a compact map of every component so the coding agent can implement without ambiguity. Read this before touching any other section.

**What the system is:** A skill memory graph sitting alongside an LLM-based agent, with **two distinct action tiers**. At episode start, the agent selects one $d=1$ strategic scaffold (an option, held fixed for the whole episode) that conditions reasoning/context. At every step within the episode, the agent selects a tactical skill ($d=2$ or $d=3$) and executes it via `env.step`. Tactical skills update via single-step Q-learning; the strategic scaffold updates once per episode via option-style return credit. Separately, the tactical layer grows (Gate 1→2 formation), shrinks (decay + pruning), and reorganizes (float-up $d=3 \to d=2$, demotion). The strategic layer ($d=1$) grows only through periodic **sleep consolidation** events — never through per-node float-up.

**Key data structures:**
- `SkillNode` — the primary data object. One per skill, at any depth. Defined in §6.3. Strategic ($d=1$) and tactical ($d=2/3$) nodes share the same class but populate different fields (see §6.3 notes).
- `CandidateRecord` — a lightweight pre-graph accumulator for **tactical** skill formation only. Lives in `candidate_pool`. Defined in §4.2.
- `SkillGraph` — a tree over `SkillNode` objects. Owns the `children_index`. Defined in §6.1.
- `EpisodicMemoryBank` — separate store of raw experiences. Linked from nodes via `evidence_ids`. Not part of the graph.

**Execution order per episode:**
1. Classify task type → `t_k`
2. **Strategic selection (once):** select $d=1$ scaffold via option-value retrieval (§3.7); hold fixed for the episode
3. For each step: tactical retrieval → select skill ($d=2/3$) → execute → compute TD error → update tactical Q → run Gate 1
4. End of episode: update strategic option-value $Q^\Omega$ over full episode return (§3.7) → run Gate 2 (candidate → tactical node) → run graph maintenance (decay + pruning + float-up, $d=3\to d=2$ only) → recompute `decay_rate` for active nodes → check sleep-consolidation trigger (§6.6)

**What MemRL provides vs. what this adds:**
- MemRL provides: basic memory bank, retrieval by similarity, episode-level updates
- This adds: hierarchical graph structure, two-tier action space (strategic options + tactical actions), gated tactical formation pipeline, transferability scoring, sleep-consolidation for strategic scaffold formation, depth-indexed decay with global salience denominator, bidirectional float-up/demotion with hysteresis

**Critical invariants the coding agent must preserve:**
- `decay_rate` on a tactical node always equals `λ_d / (Q̄_w + ε)`. Recompute after every Q-update to an active node. Strategic ($d=1$) nodes always have `decay_rate = 0.0`.
- `children_index` lives on the graph object, not the node. Node stores only `parent_id`.
- `total_accessed` = `sum(self.n.values())`. Expose as a `@property`, never store separately.
- All new **tactical** nodes enter at `depth = 3`. No exceptions. Strategic nodes are only ever created by sleep consolidation, directly at `depth = 1`.
- The pruning loop uses `node.decay_rate` directly — no `t_k` dependency in pruning. Never prunes $d=1$ nodes.
- Float-up (§6.4) only operates $d=3 \to d=2$. There is no per-node float-up into $d=1$.
- Strategic Q-values (`Q_omega`) and tactical Q-values (`Q`) are **separate dicts and must never be merged**. Mixing single-step and full-episode-return values in one dict corrupts the shrinkage-weighted mean.
- Bootstrap phase: until $K_{\text{bootstrap}}$ distinct task types have been observed (or the corresponding $d=2$ count, see §6.6.1), the strategic layer is seeded manually/by LLM reflection, not via consolidation.

---

## 1. Introduction and Motivation

Standard agent memory systems conflate several distinct questions:

- Which experience is worth storing?
- Which stored experience is worth keeping?
- Which kept experience generalizes to new tasks?

Most prior work (MemGPT, A-MEM, Voyager, SkillLib) optimizes retrieval — choosing what to surface at inference time — but treats memory formation and consolidation as secondary. This work inverts the priority: the primary contribution is a principled answer to **which experiences should consolidate into reusable skills, and at what level of generality**.

The central hypothesis is that not all useful skills are transferable. A skill may exhibit high utility within a single task distribution while remaining highly task-specific. Organizing memory by estimated transferability — rather than recency, retrieval frequency, or semantic similarity — produces a graph whose structure reflects genuine generalizability rather than usage patterns.

**Key design decisions confirmed for Phase 1:**

- A single unified hierarchical graph. Depth encodes both transferability and **role**: $d=1$ is structurally distinct from $d=2/3$, not merely "more general."
- **Two action tiers.** $d=1$ nodes are strategic scaffolds — selected once per episode, held fixed, updated via option-style (semi-MDP) return credit. $d=2/3$ nodes are tactical skills — selected every step, updated via single-step Q-learning.
- Tactical skills are formed through a two-gate pipeline (Gate 1: TD error novelty → Gate 2: utility pre-filter). Transferability gating (Gates 3–4) governs float-up **from $d=3$ to $d=2$ only**, not formation, and not entry into $d=1$.
- Strategic scaffolds ($d=1$) are formed through a separate **sleep consolidation** mechanism (§6.6): periodic batch clustering of unabsorbed $d=2$ nodes, hybrid absorb-or-spawn. This is not float-up and does not use Gates 1–4.
- Utility is estimated via Q-learning with TD updates, stored **per task type**, for tactical nodes. This per-type granularity is required for transferability scoring. Strategic nodes store a **separate** option-value $Q^\Omega$, updated once per episode.
- Memory decay uses **global weighted-mean utility** $\bar{Q}_{i,w}$ in the denominator for tactical nodes — not task-specific $Q_i(t_k)$ — because the graph is unified and decay governs global graph membership, not per-task relevance. $d=1$ nodes never decay.
- Two memory management mechanisms coexist: a decay-based pruning threshold $\theta_{\text{prune}}$ (soft) and a hard action space cap $|A| \leq N$ (hard). Both apply to the tactical layer only.

**Explicitly deferred to Phase 2:**

- Affect/personalization graph
- DAG extension for multi-parent nodes
- Memory-quality bonus term in reward
- Double Q-learning for overestimation bias correction

---

## 2. Problem Formulation

### 2.1 MDP Definition

$$\mathcal{MDP} = \left(S,\ A^{\Omega},\ A^{\tau},\ P,\ R,\ \gamma,\ \mathcal{M}\right)$$

The memory bank $\mathcal{M}$ is a **side-channel** that conditions the policy. It is not part of the state space $S$. The action space is **partitioned** into two tiers: $A^{\Omega}$ (strategic options, drawn from $d=1$) and $A^{\tau}$ (tactical actions, drawn from $d=2/3$). This is a semi-MDP over $A^{\Omega}$ nested around a standard MDP over $A^{\tau}$ — the formalism follows **Sutton, Precup & Singh's Options framework (1999)**. Embedding $\mathcal{M}$ in $S$ would make the state space grow with every new skill, creating a non-stationary MDP with no convergence guarantees.

### 2.2 State

$$s_t = \left(t_k,\ c_t,\ h_t,\ \omega\right)$$

| Component | Description |
|---|---|
| $t_k$ | Task type. Fixed within an episode. Changes between episodes. |
| $c_t$ | Task context at step $t$ — the current problem being solved. |
| $h_t$ | Short-term interaction history over the last $w$ steps. |
| $\omega$ | The active strategic scaffold for this episode. Selected once at $t=0$, fixed for the episode's duration. |

Task type $t_k$ is the primary conditioning variable for utility estimation and tactical retrieval. Its formal definition — benchmark-derived, cluster-assigned, or hierarchical taxonomy — is an open problem noted in §11.

### 2.3 Action

Two action types at two cadences:

$$a_0^{\Omega} = \omega \in \mathcal{G}_{d=1} \qquad \text{selected once, at } t=0 \text{ of the episode, held fixed}$$

$$a_t^{\tau} = s_i \in \mathcal{G}_{d \in \{2,3\}} \qquad \text{selected at every step } t \geq 0$$

The strategic action $\omega$ does **not** produce an environment transition directly — it conditions the agent's reasoning context for the remainder of the episode (modifying how $c_t$/$h_t$ are interpreted, not what `env.step` receives). The tactical action $a_t^\tau$ is what is passed to `env.step`. Token-level generation remains outside this MDP.

### 2.4 Transition

$$s_{t+1} = \left(t_k,\ c_{t+1},\ h_{t+1},\ \omega\right)$$

$c_{t+1}$ reflects the outcome of applying tactical action $a_t^\tau$. History updates as $h_{t+1} = h_t \cup \{(a_t^\tau, r_t)\}$. $\omega$ is invariant across the episode and does not appear in the transition update; task type $t_k$ is likewise invariant within an episode.

### 2.5 Reward

$$r_t = r_t^{\text{env}}$$

Environment feedback per reasoning step, attributed to the active tactical action $a_t^\tau$. A memory-quality bonus is defined but deferred:

$$r_t^{\text{full}} = r_t^{\text{env}} + \beta \cdot r_t^{\text{mem}}, \qquad \beta = 0 \text{ in Phase 1}$$

The strategic action $\omega$ does not receive $r_t$ directly — it receives episode-level credit via the option-value update in §3.7.

### 2.6 Discount Factor

$$\gamma \in [0.9,\ 0.99]$$

Shared between tactical TD updates (§3.2) and the strategic option-value update (§3.7). High because a skill deployed now may enable better subsequent skills, and because the strategic scaffold's value depends on the full discounted episode return.

### 2.7 Memory Bank

$$\mathcal{M}_t = \left(\mathcal{G}_t,\ \{Q_i(t_k)\},\ \{Q^{\Omega}_j(t_k)\},\ \{\lambda_d\},\ \epsilon\right)$$

| Component | Description |
|---|---|
| $\mathcal{G}_t$ | Unified skill graph at time $t$, containing both strategic ($d=1$) and tactical ($d=2/3$) nodes |
| $\{Q_i(t_k)\}$ | Tactical Q-estimates, indexed by skill and task type |
| $\{Q^{\Omega}_j(t_k)\}$ | Strategic option-value estimates, indexed by scaffold and task type. **Stored separately from** $\{Q_i(t_k)\}$ — never merged into the same dict. |
| $\{\lambda_d\}$ | Depth-indexed base decay rates (not baking in Q). $\lambda_1 = 0$ always. |
| $\epsilon$ | Utility floor for decay denominator |

$\mathcal{M}$ is updated **after each episode**, not per step. Tactical Q-updates happen per step but are written to the graph at episode end alongside the strategic update; both are part of the same end-of-episode commit in the reference implementation (§10).

---

## 3. Utility Estimation

This section covers **tactical** utility (§3.1–3.6, $d=2/3$, single-step Q-learning). Strategic utility ($d=1$, option-value, once-per-episode) is covered separately in §3.7, since it uses a different update rule and a different Q-store, never to be merged with the tactical one.

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
| $\mathcal{N}(s_i)$ | Local neighborhood: parent node + child nodes of $s_i$, restricted to $d \in \{2,3\}$. |

This is Q-learning (off-policy): the update uses the greedy policy over the local neighborhood regardless of which skill was actually selected.

After a Q-update to node $s_i$, immediately call `s_i.recompute_decay_rate()` (see §7.4). This keeps `decay_rate` fresh at the cost of one arithmetic operation per active node per episode.

### 3.3 TD Error

$$\delta_t = r_t + \gamma \max_{s_j \in \mathcal{N}(s_i)} Q_j(t_k) - Q_i(t_k)$$

$\delta_t$ drives the tactical Q-update and serves as the novelty signal for Gate 1. This signal does not apply to the strategic tier (see §3.7 for why).

### 3.4 Initialization

$$Q_i(t_k) = 0 \quad \forall\, t_k$$

No utility prior at creation. Applies to tactical nodes at formation (§4.3). Strategic nodes use a separate initialization rule (§3.7).

### 3.5 Tactical Action Selection

$$a_t^{\tau} = \arg\max_{s_i \in \text{top-}k}\ Q_i(t_k)$$

Candidates are shortlisted by semantic similarity before Q-ranking, restricted to $d \in \{2,3\}$. Retrieval technique is an open decision (§11).

### 3.6 Failure Credit Assignment

When a negative surprise ($\delta_t < -\theta_\delta$) occurs at step $t$, the penalty is distributed across all **tactical** skills active in the episode using **recency-weighted credit**:

$$\Delta Q_{s}(t_k) = -|\delta_t| \cdot \gamma^{T - \text{step}(s)}$$

where $T$ is the current step and $\text{step}(s)$ is the step at which skill $s$ was last active. Skills used most recently before the failure receive the largest penalty; earlier skills receive exponentially smaller corrections. This prevents good setup-skills from being uniformly penalized for a failure they did not cause. The active strategic scaffold $\omega$ is not penalized by this mechanism — its credit is resolved separately, once, at episode end (§3.7), since it is not meaningful to attribute a single step's surprise to a choice made at $t=0$ for the whole episode.

> **Implementation note:** Maintain `active_skills: list[tuple[SkillNode, int]]` — tuples of (node, step_index), tactical nodes only — during each episode. On negative Gate 1 trigger, iterate this list and apply the weighted penalty.

### 3.7 Strategic Option-Value Update

The active scaffold $\omega$ is selected once at $t=0$ and held fixed for the episode (confirmed design decision). Its value is updated **once, at episode end**, using the full discounted episode return — not a per-step TD target. This is the standard option-value update from the Options framework (Sutton, Precup & Singh, 1999), specialized to the case where the option always runs to episode termination (no early termination/interruption in Phase 1, so there is no bootstrap term across option boundaries):

$$Q^{\Omega}_{\omega}(t_k)\ \leftarrow\ Q^{\Omega}_{\omega}(t_k)\ +\ \alpha^{\Omega} \Bigl[\sum_{t=0}^{T-1} \gamma^t r_t\ -\ Q^{\Omega}_{\omega}(t_k)\Bigr]$$

| Term | Description |
|---|---|
| $\alpha^{\Omega}$ | Strategic learning rate. May differ from tactical $\alpha$; starting value $0.1$, swept independently. |
| $\sum_{t=0}^{T-1} \gamma^t r_t$ | Full discounted return over the episode under scaffold $\omega$. |
| $T$ | Episode length (number of steps). |

**Why no per-step bootstrap term:** Standard option-value updates include a term $\gamma^T \max_{\omega'} Q^{\Omega}_{\omega'}(t_k)$ to bootstrap past the option's end when options can terminate early and chain within a longer trajectory. Phase 1 fixes the scaffold for the entire episode (confirmed decision), so there is no "past the option" state within the episode boundary — the sum already covers the full trajectory the option was responsible for. This term is reintroduced if Phase 2 allows mid-episode re-selection.

**Storage:** $Q^{\Omega}_{\omega}(t_k)$ is stored in a separate dict on the node, never merged with tactical `Q`. See §6.3 schema notes — strategic nodes populate `Q_omega`, tactical nodes populate `Q`; a given node only ever populates one of the two, determined by its depth.

**Initialization:** $Q^{\Omega}_{\omega}(t_k) = 0\ \forall t_k$ at scaffold creation (via sleep consolidation, §6.6), matching the tactical no-prior convention.

**Why Gate 1 does not apply to strategic candidates:** Gate 1 requires a per-step TD error, which presupposes a per-step reward signal. The strategic tier only receives a reward signal once per episode (the full return), so there is no analogous per-step novelty quantity to gate on. Strategic scaffold formation is therefore handled entirely by the sleep-consolidation mechanism (§6.6), which uses clustering and embedding similarity rather than a TD-error admission gate.

---

## 4. Gated Skill Formation Pipeline (Tactical Layer Only)

Raw experiences do not directly become skill nodes. A two-gate formation pipeline controls admission and node creation **for the tactical layer ($d=2/3$)**. Transferability gates (3–4) govern float-up from $d=3$ to $d=2$ separately and are documented in §6.4. Strategic ($d=1$) scaffold formation does not use this pipeline at all — see §6.6.

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

## 5. Transferability — Full Formalism (Tactical Layer Only)

Transferability scoring governs float-up from $d=3$ to $d=2$ only. It is never used for pruning, formation, or strategic ($d=1$) scaffold construction — those use sleep consolidation (§6.6) instead.

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

Depth now encodes both **role** and (within the tactical layer) **transferability**. $d=1$ is not "the highest transferability band reached by float-up" — it is a structurally distinct population, entered only via sleep consolidation (§6.6), never via per-node float-up. $d=2/3$ retain the original transferability-threshold semantics for tactical skills.

$$\text{depth}(s_i) = \begin{cases} 0 & \text{virtual root — structural anchor only} \\ 1 & \text{strategic scaffold — entered only via sleep consolidation (§6.6), never by float-up} \\ 2 & \hat{T}(s_i) \geq \theta_1 \quad \text{tactical, semi-general — reached by float-up from } d=3 \\ 3 & \hat{T}(s_i) < \theta_1 \quad \text{tactical, task-specific (leaf)} \end{cases}$$

**Starting threshold:** $\theta_1 = 0.75$, swept in ablation. (The previous three-band tactical scale — $\theta_1, \theta_2$ — collapses to a single float-up threshold $\theta_1$ now that $d=1$ is no longer a transferability band; $\theta_2$ is retired. See §12.)

**Invariant (tactical):** All new tactical nodes enter at $d = 3$. Depth can only decrease (float up) to $d=2$ as evidence accumulates — never directly to $d=1$, and never increases on creation.

**Invariant (strategic):** All new strategic nodes are created directly at $d=1$ by sleep consolidation. No node is ever created at $d=1$ by any other mechanism, and no $d=2$ node transitions to $d=1$ by float-up.

### 6.2.1 Bootstrap Phase

Before the tactical layer has produced enough material to consolidate, $d=1$ is empty or sparse. Define $K_{\text{bootstrap}}$, tied to the same evidence-grounding convention used elsewhere in this spec (cf. $N_{\min} = 5K$): the bootstrap phase ends once the unabsorbed $d=2$ population (§6.6.1) first reaches $N_{\text{sleep}}$, the sleep-trigger threshold. Until then:

- $d=1$ is seeded manually or via a one-time LLM reflection pass over early task types, not via consolidation.
- The agent may operate with zero or a small fixed number of strategic scaffolds during bootstrap; strategic selection degrades gracefully to "no scaffold" (null $\omega$) if $d=1$ is empty.
- No formal gating applies to bootstrap-seeded nodes; they are simply present, and become subject to normal sleep-consolidation absorption logic (§6.6) once regular consolidation begins.

### 6.3 Node Schema — FINALIZED for Phase 1

`SkillNode` is shared across both tiers. A given instance populates **either** the tactical fields (`Q`, `n`) **or** the strategic fields (`Q_omega`, `n_omega`), determined by `depth` — never both. This is enforced by convention, not by separate classes, to keep the graph homogeneous for traversal; the coding agent must not write to the tactical dict on a $d=1$ node or vice versa.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SkillNode:
    # --- Identity ---
    id: str                          # UUID, assigned at creation

    # --- Skill Representation ---
    content: str                     # LLM-generated procedural summary (tactical) or
                                     # synthesized strategic framing (strategic, from sleep
                                     # consolidation — see §6.6)
    embedding: np.ndarray            # Dense vector; used for retrieval ranking and
                                     # parent-finding on float-up / cluster absorption

    # --- Provenance ---
    task_type_primary: str           # Task type t_k under which skill was first formed.
                                     # For strategic nodes: task type that dominated the
                                     # cluster that produced this scaffold.
    t_create: int                    # Global retrieval step at creation

    # --- Hierarchy ---
    depth: int                       # Current depth ∈ {1, 2, 3}. Tactical nodes always
                                     # 3 at creation; strategic nodes always 1 at creation.
    parent_id: str | None            # UUID of parent node. None only for virtual root.
    secondary_parents: list[str] = field(default_factory=list)
                                     # Reserved for Phase 2 DAG extension. Empty in Phase 1.

    # --- Usage Statistics ---
    last_accessed_step: int = 0      # Global step index of most recent retrieval.
                                     # Used to compute Δt in decay formula (tactical only;
                                     # strategic nodes never decay, see decay_rate below).

    # --- Tactical Utility Tracking (d=2/3 ONLY) ---
    Q: dict[str, float] = field(default_factory=dict)
                                     # Q(t_k): per-task-type Q-values, single-step TD (§3.2).
                                     # Empty/unused on strategic nodes.
    n: dict[str, int] = field(default_factory=dict)
                                     # n(t_k): retrieval counts per task type.
                                     # Used for shrinkage weights and Gate 3.
                                     # Empty/unused on strategic nodes.

    # --- Strategic Option-Value Tracking (d=1 ONLY) ---
    Q_omega: dict[str, float] = field(default_factory=dict)
                                     # Q^Ω(t_k): per-task-type option values, updated once
                                     # per episode over the full discounted return (§3.7).
                                     # SEPARATE dict from Q — never merge. Empty/unused on
                                     # tactical nodes.
    n_omega: dict[str, int] = field(default_factory=dict)
                                     # Episode count this scaffold was selected, per task type.
                                     # Empty/unused on tactical nodes.

    # --- Retention ---
    decay_rate: float = 0.0          # Cached value of λ_d / (Q̄_w + ε) for tactical nodes.
                                     # GLOBAL salience denominator — uses weighted-mean
                                     # utility across all task types, NOT task-specific Q.
                                     # Recomputed after every Q-update via recompute_decay_rate().
                                     # Strategic (d=1) nodes: always 0.0, never decay.

    # --- Episodic Links ---
    evidence_ids: list[str] = field(default_factory=list)
                                     # IDs into the EpisodicMemoryBank.
                                     # Capped at R entries via reservoir sampling.
                                     # Provides diagnostic trace back to raw experiences.

    # --- Sleep Consolidation Bookkeeping (d=2 ONLY) ---
    absorbed_by_sleep: bool = False  # Set True the moment a sleep event (§6.6) assigns this
                                     # d=2 node to a d=1 parent (whether by absorption into an
                                     # existing scaffold or as part of a newly-spawned cluster).
                                     # Drives the sleep-trigger counter (§6.6.1). Meaningless
                                     # on d=1 and d=3 nodes; always False there.

    # --- Derived Properties ---
    @property
    def total_accessed(self) -> int:
        """Total tactical retrievals across all task types. Derived — never stored
        separately. Undefined/zero for strategic nodes (use n_omega instead)."""
        return sum(self.n.values())

    def recompute_decay_rate(self, lambda_d: float, epsilon: float) -> None:
        """
        Recompute and cache the global decay rate. TACTICAL NODES ONLY.

        decay_rate = λ_d / (Q̄_w + ε)

        where Q̄_w is the confidence-weighted mean utility across ALL task types,
        computed from the tactical Q dict. This is the GLOBAL salience denominator —
        consistent with the unified (non-partitioned) graph.

        Call this after every Q-update to any task type on this node.
        At d=1 (strategic), this method should not be called in normal operation;
        if called, it sets decay_rate = 0.0 unconditionally as a safety fallback.
        """
        if self.depth == 1:
            self.decay_rate = 0.0
            return
        Q_bar_w = self._weighted_mean_utility(lambda_shrink=10)
        self.decay_rate = lambda_d / (Q_bar_w + epsilon)

    def _weighted_mean_utility(self, lambda_shrink: float = 10) -> float:
        """Bayesian shrinkage weighted mean over the TACTICAL Q dict:
        Q̄_w = Σ w_ik Q(t_k) / Σ w_ik. Not applicable to Q_omega."""
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
| `content` | Open: raw reasoning trace vs. distilled procedural summary (tactical). For strategic nodes, content is LLM-synthesized from a cluster of $d=2$ skill descriptions during sleep consolidation (§6.6.2) — a different prompt than tactical extraction. |
| `embedding` | Open: frozen LLM encoder vs. fine-tuned. Populated at node creation. For strategic nodes created by consolidation, may be the cluster centroid embedding or a fresh embedding of the synthesized content — pick one and document it; do not leave ambiguous. |
| `decay_rate` | Tactical: always equals `λ_d / (Q̄_w + ε)`. Strategic: always `0.0`. Never compute retention inline without calling `recompute_decay_rate()` first. |
| `Q` vs `Q_omega` | Mutually exclusive by depth. A coding-time assertion (`assert depth == 1 or not Q_omega`, `assert depth != 1 or not Q`) is recommended to catch accidental cross-writes early. |
| `evidence_ids` | Reservoir-sampled. Implement `add_evidence(eid)` with reservoir sampling at cap $R$. $R$ is a hyperparameter (suggested starting: 50). |
| `absorbed_by_sleep` | Only meaningful on $d=2$ nodes. Drives the unabsorbed-count sleep trigger (§6.6.1) — do not repurpose for any other bookkeeping. |
| `secondary_parents` | Do not read or write in Phase 1. Initialize empty. |
| `total_accessed` | A `@property`. Do not add a stored counter — it will diverge. |

### 6.4 Float-Up Mechanism (Gates 3 and 4) — $d=3 \to d=2$ ONLY

Float-up is now a **single-threshold, single-transition** mechanism: a $d=3$ tactical node either qualifies for $d=2$ or it doesn't. There is no float-up target beyond $d=2$; entry into $d=1$ is handled exclusively by sleep consolidation (§6.6).

```python
def maybe_float_up(node: SkillNode, graph: SkillGraph, K: int,
                   N_min: int, theta_CV: float,
                   theta_1: float, epsilon_hyst: float) -> None:
    # Only applies to d=3 tactical nodes. d=2 and d=1 nodes never call this.
    if node.depth != 3:
        return

    # Gate 3: Confidence — sufficient cross-task evidence
    if node.total_accessed < N_min:          # N_min = 5K
        return

    # Gate 4: Stability — CV of utility across task types
    cv = compute_cv(node)                    # sqrt(Var_w) / Q̄_w
    if cv >= theta_CV:
        return

    # Compute transferability score
    T_hat = compute_transferability(node)    # Q̄_w² / (Q̄_w² + Var_w)

    if T_hat >= theta_1:
        # Reparent: find highest cosine-similarity node at depth 1... NO — at depth 2.
        # Float-up never targets d=1 directly. The new parent is the best-matching
        # existing d=2 node, or a freshly created d=2 "bucket" node if no suitable
        # d=2 parent exists yet (bootstrap case for the d=2 layer itself).
        new_parent = graph.find_best_parent(node, target_depth=2)
        graph.reparent(node, new_parent)     # atomic: updates children_index + parent_id
        node.depth = 2
        node.absorbed_by_sleep = False       # newly arrived at d=2; eligible for next sleep event
        node.recompute_decay_rate(graph.lambda_d[node.depth], graph.epsilon)
```

`find_best_parent` uses cosine similarity over `embedding`. This is one of two structural uses of semantic similarity for graph rewiring — the other is cluster-centroid-to-scaffold matching during sleep consolidation (§6.6.2). Retrieval similarity (§9) is a separate, unrelated use for ranking only.

**Important:** a node arriving at $d=2$ via float-up starts with `absorbed_by_sleep = False`. This is what makes it visible to the sleep-trigger counter (§6.6.1) and eligible for the next consolidation event.

### 6.5 Demotion and Hysteresis ($d=3 \leftrightarrow d=2$ ONLY)

Demotion, like float-up, now only ever moves a node between $d=2$ and $d=3$. There is no demotion path out of $d=1$ in Phase 1 — strategic scaffolds are permanent once created by consolidation (consistent with $\lambda_d = 0$ at $d=1$; nothing in Phase 1 removes or downgrades a strategic node).

$$\text{demote}(s_i) \iff \hat{T}(s_i) < \theta_1 - \epsilon_{\text{hyst}}, \qquad d(s_i) = 2$$

where $\epsilon_{\text{hyst}} = 0.1$. A demoted node returns to $d=3$ and its `absorbed_by_sleep` flag is irrelevant again (reset to `False` for cleanliness, though it has no effect at $d=3$).

A recently promoted node is exempt from demotion for $M_{\text{wait}}$ episodes (grace period, prevents oscillation).

### 6.6 Sleep Consolidation — Strategic Scaffold Formation

This is the **only** mechanism by which $d=1$ nodes are created or grow children, after the bootstrap phase (§6.2.1). It is a periodic, batch process — not a per-node, continuous one like float-up. It is triggered by an evidence-accumulation counter (§6.6.1), clusters the eligible $d=2$ population, and either absorbs each resulting cluster into an existing scaffold or spawns a new one (§6.6.2).

#### 6.6.1 Trigger Condition

The trigger counter tracks **unabsorbed $d=2$ nodes**, not raw live $d=2$ population:

$$\text{count}_{\text{unabsorbed}} = |\{\, n \in \mathcal{G}_{d=2} : \neg n.\texttt{absorbed\_by\_sleep} \,\}|$$

A sleep event fires when:

$$\text{count}_{\text{unabsorbed}} \geq N_{\text{sleep}}$$

**Why not raw live $d=2$ population:** Population fluctuates due to pruning (§8.1) removing low-retention $d=2$ nodes independent of consolidation activity. A raw-population counter can re-cross a fixed threshold purely from churn among already-absorbed nodes, triggering redundant consolidation passes over material that's already been abstracted. Gating on the unabsorbed-only count means the trigger only fires in response to genuinely new, unprocessed material — nodes that floated up from $d=3$ since the last sleep event and haven't yet been clustered. Pruning a node that happens to be unabsorbed simply removes it from the count without ever consuming a sleep event on it; pruning an absorbed node has no effect on the counter at all, since absorbed nodes are excluded regardless of liveness.

**Starting value:** $N_{\text{sleep}}$ is a new hyperparameter (§12), swept in ablation.

#### 6.6.2 Consolidation Procedure

```python
def sleep_consolidation(graph: SkillGraph, theta_absorb: float) -> None:
    unabsorbed = [n for n in graph.nodes_at_depth(2) if not n.absorbed_by_sleep]
    if not unabsorbed:
        return

    # Step 1: cluster the unabsorbed set only (never re-cluster already-absorbed nodes)
    clusters = cluster_embeddings([n.embedding for n in unabsorbed])  # e.g. HDBSCAN

    for cluster in clusters:
        centroid = mean_embedding(cluster)

        # Step 2: check absorption against EXISTING d=1 scaffolds
        existing_d1 = graph.nodes_at_depth(1)
        if existing_d1:
            best_parent, similarity = max(
                ((p, cosine_sim(centroid, p.embedding)) for p in existing_d1),
                key=lambda x: x[1]
            )
        else:
            similarity = -1.0  # forces spawn if no d=1 nodes exist yet

        if similarity >= theta_absorb:
            # Absorb: reparent the whole cluster under the existing scaffold
            for node in cluster:
                graph.reparent(node, best_parent)   # children_index updated atomically
                node.absorbed_by_sleep = True
            # Optionally update best_parent.embedding as a running average — pick one
            # policy and document it; do not leave ambiguous.
        else:
            # Spawn: synthesize a new d=1 scaffold from this cluster
            content = llm_synthesize_scaffold([n.content for n in cluster])  # new prompt,
                                                                              # distinct from
                                                                              # tactical extraction
            new_scaffold = SkillNode(
                id=new_uuid(), content=content, embedding=centroid,
                task_type_primary=majority_task_type(cluster),
                t_create=graph.current_step, depth=1, parent_id=graph.root.id,
                Q_omega={}, n_omega={}, decay_rate=0.0,
            )
            graph.insert(new_scaffold, parent=graph.root)
            for node in cluster:
                graph.reparent(node, new_scaffold)
                node.absorbed_by_sleep = True
```

**Ordering constraint:** sleep consolidation must run strictly after all per-node float-up resolution in a given maintenance pass (§10), and the two must never run concurrently against the same node set. Float-up writes `parent_id`/`children_index` for $d=3 \to d=2$ transitions; consolidation writes them for $d=2 \to d=1$ transitions. Sequencing float-up first, then checking the sleep trigger, then running consolidation if triggered, avoids the node being mid-evaluation in one process while reassigned by the other.

**New hyperparameter:** $\theta_{\text{absorb}}$ — minimum cluster-centroid-to-scaffold cosine similarity for absorption rather than spawning a new scaffold. Added to §12.

**Open (clustering method):** `cluster_embeddings` algorithm (HDBSCAN vs. k-means vs. agglomerative) and its parameters (e.g. minimum cluster size, distance metric) are not finalized — added to §11 open problems. The trigger condition and absorb/spawn decision rule are confirmed regardless of which clustering method is chosen underneath.

---

## 7. Memory Decay (Tactical Layer Only — $d=2/3$)

### 7.1 Design Rationale: Global vs. Task-Local Decay

This entire section applies to the **tactical layer** ($d=2/3$). Strategic ($d=1$) nodes do not decay at all — not because their global utility happens to be high, but because they are a structurally permanent population, entered only via sleep consolidation and never removed in Phase 1. This is stronger than "zero decay rate as a special case"; $d=1$ nodes are categorically outside the decay/pruning mechanism described below.

The tactical graph is **unified** — there is no separate graph per task type. Decay governs global graph membership: whether a tactical node survives at all. It is therefore architecturally incoherent to decay a node using task-local $Q_i(t_k)$, which would make the same node's retention path-dependent on whatever task type happened to run last.

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

### 8.2 Mechanism 2 — Tactical Action Space Cap

$$|A^\tau| \leq N$$

At tactical retrieval time, only the top-$N$ nodes per depth ($d=2$ and $d=3$ counted separately) by current Q-value are eligible. This is a hard computational guarantee on the tactical action space size, required for Q-learning convergence (fixed finite action space assumption). It does not apply to strategic ($d=1$) selection, which happens once per episode over a much smaller population by construction (sleep consolidation keeps $d=1$ small) and does not face the same convergence pressure from a growing action space.

**Interaction with decay:** Decay governs whether a tactical node *exists* in the graph. The $N$-cap governs whether an existing tactical node is *reachable* by the retrieval policy. A node can exist but be excluded from the action space if it ranks outside top-$N$ at its depth.

| Mechanism | Type | Controls | Hyperparameter |
|---|---|---|---|
| Ebbinghaus decay + $\theta_{\text{prune}}$ | Soft, continuous | Graph membership | $\theta_{\text{prune}}$ |
| $|A| \leq N$ | Hard, discrete | Retrieval eligibility | $N$ |

---

## 9. Retrieval

> ⚠️ **OPEN — NOT FINALIZED.** Confirmed constraints below for both tiers. Exact tactical shortlist method (ANN vs. BM25 vs. hybrid) is the one remaining open design decision. Strategic retrieval is structurally simpler and fully specified.

Retrieval is now two separate procedures at two cadences, not one ranked list. They draw from disjoint depth ranges and are never merged into a single top-$k$.

### 9.1 Strategic Retrieval (Once Per Episode, $d=1$ only)

At $t=0$ of each episode, select the scaffold $\omega$ for the task type $t_k$:

$$\omega = \arg\max_{\omega_j \in \mathcal{G}_{d=1}} Q^{\Omega}_{\omega_j}(t_k)$$

Since $\mathcal{G}_{d=1}$ is small by construction (sleep consolidation keeps it bounded — typically tens, not thousands, of nodes), this is a full scan, not a shortlist-then-rerank pipeline. No embedding similarity step is required at this tier; the choice is driven entirely by accumulated option-value evidence. If $\mathcal{G}_{d=1}$ is empty (bootstrap phase, §6.2.1), $\omega = \text{null}$ and the episode proceeds with no strategic conditioning.

**Cold task type:** if $Q^{\Omega}_{\omega_j}(t_k)$ is undefined for all $\omega_j$ (task type never seen at the strategic tier), fall back to the scaffold with highest global mean $\bar{Q}^{\Omega}_{\omega_j}$ across all task types it has been used on — the same shrinkage-weighted-mean construction as §5.3, applied to $Q_\omega$ instead of $Q$.

### 9.2 Tactical Retrieval (Every Step, $d=2/3$ only)

#### 9.2.1 Confirmed Constraints

1. **Bottom-up traversal.** Start at $d = 3$ leaf nodes matching $t_k$, walk ancestor chains toward $d = 2$. Traversal terminates at $d=2$ — it does not continue into $d=1$, since $d=1$ is not part of the tactical action space.

2. **Retrieval score formula:**

$$\text{score}(s_i,\ \Delta t) = d_i(\Delta t) \cdot \cos(e_i,\ e_q)$$

   where $d_i(\Delta t) = e^{-\text{decay\_rate} \cdot \Delta t}$ (uses cached `decay_rate`) and $\cos(e_i, e_q)$ is cosine similarity between skill embedding and query embedding.

3. **Action space bound respected.** Only top-$N$ tactical nodes per depth eligible (§8.2).

4. **Traversal cost:** $O(m \cdot D')$ where $m$ = shortlist size, $D' = 2$ (depths 2 and 3 only — one fewer level than before, since $d=1$ is excluded from this traversal).

#### 9.2.2 Conceptual Direction

Leading candidate: **bottom-up semantic search with ancestor expansion, restricted to $d \in \{2,3\}$.**

- Step 1: Semantic shortlist of $m$ leaf nodes at $d = 3$ tagged with $t_k$
- Step 2: Collect ancestor chains of all shortlisted nodes, up to and including their $d=2$ parent only
- Step 3: Score all candidates by retrieval score formula
- Step 4: Return top-$k$, subject to the $|A^\tau| \leq N$ cap (§8.2)

The exact method for Step 1 — embedding nearest-neighbor search, BM25, or hybrid — remains open (§11).

---

## 10. Episode Update Loop

```python
# G               — SkillGraph object
# candidate_pool  — dict[str, CandidateRecord]
# current_step    — global retrieval step counter
# active_skills   — list of (SkillNode, step_index) tuples, TACTICAL nodes only, for credit assignment
# episode_rewards — list of r_t, accumulated across the episode, for the strategic update

for each episode:
    t_k = classify_task(episode)
    active_skills = []
    episode_rewards = []

    # --- STRATEGIC SELECTION (once, t=0) ---
    omega = select_strategic_scaffold(G, t_k)   # §9.1; null if G has no d=1 nodes yet (bootstrap)

    for each step t in episode:

        # TACTICAL RETRIEVAL (§9.2) — conditioned on omega via prompt/context, not via Q
        candidates = recall_tactical(query=c_t, task_type=t_k, k=5)   # d ∈ {2,3} only
        a_t = argmax_{s_i in candidates} Q_i[t_k]
        active_skills.append((a_t, t))

        # EXECUTION
        r_t, s_{t+1} = env.step(a_t)
        episode_rewards.append(r_t)

        # TACTICAL TD UPDATE
        delta_t = r_t + gamma * max_{s_j in N(a_t)} Q_j[t_k] - Q_{a_t}[t_k]
        Q_{a_t}[t_k] += alpha * delta_t
        a_t.n[t_k] = a_t.n.get(t_k, 0) + 1
        a_t.last_accessed_step = current_step
        a_t.recompute_decay_rate(lambda_d[a_t.depth], epsilon)  # update global salience cache
                                                                  # (tactical nodes only; no-op
                                                                  # guard inside the method
                                                                  # protects against d=1 misuse)

        # GATE 1 — Experience Admission (tactical only; strategic has no per-step gate, see §3.7)
        if delta_t > theta_delta:
            candidate_pool.add_or_update(experience(s_t, a_t, r_t, t_k))
        elif delta_t < -theta_delta:
            # Recency-weighted failure credit (§3.6) — tactical skills only, omega excluded
            T = current_step
            for (s, step_s) in active_skills:
                penalty = abs(delta_t) * (gamma ** (T - step_s))
                Q_s[t_k] -= penalty
                s.recompute_decay_rate(lambda_d[s.depth], epsilon)

        current_step += 1

    # END OF EPISODE

    # --- STRATEGIC OPTION-VALUE UPDATE (once, §3.7) ---
    if omega is not None:
        episode_return = sum(gamma**t * r for t, r in enumerate(episode_rewards))
        omega.Q_omega[t_k] = omega.Q_omega.get(t_k, 0.0) + alpha_omega * (
            episode_return - omega.Q_omega.get(t_k, 0.0)
        )
        omega.n_omega[t_k] = omega.n_omega.get(t_k, 0) + 1
        # omega.decay_rate stays 0.0 — never touched

    # Gate 2 — Candidate → Tactical Node (batch LLM extraction here)
    for m_i in list(candidate_pool.values()):
        if m_i.n > N_skill and m_i.Q_mean > theta_U:
            new_node = create_skill_node(m_i)          # LLM summary + embedding call, depth=3
            parent = G.find_best_parent(new_node, target_depth=2)  # bucket parent at d=2
            G.insert(new_node, parent)
            del candidate_pool[m_i.id]

    # Graph maintenance — tactical layer, task-agnostic, no t_k dependency
    for node in list(G.nodes_at_depth(2)) + list(G.nodes_at_depth(3)):
        delta_t_node = current_step - node.last_accessed_step
        retention = exp(-node.decay_rate * delta_t_node)
        if retention < theta_prune:
            G.remove(node)
            continue
        maybe_float_up(node, G, K, N_min, theta_CV, theta_1, epsilon_hyst)   # d=3 -> d=2 only

    # Sleep-consolidation trigger check (§6.6.1) — strictly after float-up resolution above
    unabsorbed_count = sum(1 for n in G.nodes_at_depth(2) if not n.absorbed_by_sleep)
    if unabsorbed_count >= N_sleep:
        sleep_consolidation(G, theta_absorb)   # §6.6.2
```

**Note on `omega` conditioning:** the pseudocode shows `omega` passed implicitly into `recall_tactical` as context, not as a Q-influencing term. Tactical Q-values are never blended with $Q^\Omega$; the scaffold's effect on tactical selection is entirely through whatever the agent's reasoning/prompt construction does with the scaffold's `content`, not through the retrieval scoring formula in §9.2.1.

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
| $\theta_\delta$ | TD error admission threshold (tactical Gate 1) | — | sweep |
| $N_{\text{skill}}$ | Min activations for node creation (Gate 2) | — | sweep |
| $\theta_U$ | Min mean utility for node creation (Gate 2) | — | sweep |
| $N_{\min}$ | Min cross-task evidence for float-up (Gate 3) | $5K$ | derived from $K$ |
| $\theta_{\text{CV}}$ | Max coefficient of variation (Gate 4) | — | sweep |
| $\theta_1$ | Transferability cutoff for float-up $d=3 \to d=2$ | $0.75$ | sweep |
| ~~$\theta_2$~~ | ~~Transferability cutoff for $d=2$ band~~ | — | **retired** — $d=1$ is no longer a transferability band; see §6.2 |
| $\lambda$ | Bayesian shrinkage pseudocount | $10$ | sweep |
| $\lambda_{\text{slow}}$ | Base decay rate at $d = 2$ (tactical) | — | sweep |
| $\lambda_{\text{fast}}$ | Base decay rate at $d = 3$ (tactical) | $5 \times \lambda_{\text{slow}}$ | derived |
| $\epsilon$ | Utility floor in decay denominator | $0.01$ | sweep |
| $\theta_{\text{prune}}$ | Retention threshold for tactical node removal | — | sweep |
| $N$ | Hard tactical action space cap per depth ($d=2,3$) | — | sweep |
| $\alpha$ | Tactical TD learning rate | $0.1$ | sweep |
| $\alpha^{\Omega}$ | Strategic option-value learning rate | $0.1$ | sweep, independent of $\alpha$ |
| $\gamma$ | Discount factor (shared, tactical + strategic) | $[0.9, 0.99]$ | sweep |
| $\epsilon_{\text{hyst}}$ | Demotion hysteresis buffer ($d=2 \leftrightarrow d=3$) | $0.1$ | sweep |
| $M_{\text{wait}}$ | Grace period episodes after promotion | — | sweep |
| $R$ | Evidence reservoir size per node | $50$ | sweep |
| $N_{\text{sleep}}$ | Unabsorbed $d=2$ count that triggers sleep consolidation | — | sweep |
| $\theta_{\text{absorb}}$ | Min cluster-centroid-to-scaffold cosine similarity for absorption (else spawn new $d=1$ node) | — | sweep |
| $K_{\text{bootstrap}}$ | Implicit — bootstrap ends when unabsorbed $d=2$ count first reaches $N_{\text{sleep}}$ | — | derived from $N_{\text{sleep}}$ |

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