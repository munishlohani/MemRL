# Agent Memory with Utility-Based Skill Consolidation
## Architecture Specification — Phase 1
**Working Paper | Summer 2026**

---

## Abstract

We propose a memory architecture for AI agents that organizes skills within a two-tier hierarchical graph. The **strategic tier** ($d=1$) holds reasoning scaffolds — abstract frames selected once per episode under an options/semi-MDP formalism, with option-values stored per task type. The **tactical tier** (flat) holds directly executable skills formed from experience and retained via utility-modulated Ebbinghaus decay. Tactical skills are admitted by a TD error pre-filter followed by LLM judgment, stored immediately, and pruned by decay. Periodically, a **sleep consolidation** event clusters surviving tactical memories and uses LLM judgment to abstract them into strategic scaffolds — the sole mechanism by which $d=1$ nodes are created. The system is framed as a two-tier extended semi-MDP. Both tactical and strategic Q-values are stored **per task type**. Memory retention follows a biologically-grounded Ebbinghaus decay formula modulated by the **confidence-weighted mean utility** $\bar{Q}_{i,w}$ across task types — a task-agnostic salience denominator consistent with the unified (non-partitioned) graph design.

**Base template:** MemRL. This architecture extends MemRL by: (1) replacing flat memory with a two-tier hierarchical graph whose structure is determined by utility evidence and LLM abstraction rather than recency alone; (2) introducing a gated tactical formation pipeline with LLM judgment; (3) a separate sleep-consolidation pipeline for strategic scaffold formation; (4) options-style credit assignment for strategic actions; and (5) utility-modulated decay salience that governs global graph membership.

**Key departure from MemRL:** MemRL delegates all memory quality judgment to the backbone LLM's in-context reasoning at retrieval time. This architecture offloads structural decisions — what to form, what to retain, when to consolidate — to an algorithmic layer (TD error, decay, clustering), while trusting the LLM for semantic judgment (formation quality, consolidation content synthesis). The combination reduces the burden on the LLM while preserving its strength in semantic abstraction.

> **Status:** Phase 1 architecture confirmed. Node schema finalized. Q-value representation confirmed: per-task-type for both tactical and strategic nodes; decay salience uses shrinkage-weighted mean $\bar{Q}_{i,w}$. Tactical retrieval technique is one remaining open design decision.

---

## Implementation Notes for Coding Agent

This section provides a compact map of every component. Read this before touching any other section.

**What the system is:** A two-tier skill memory graph sitting alongside an LLM-based agent. At episode start, the agent selects one $d=1$ strategic scaffold (an option, held fixed for the whole episode) that conditions reasoning context. At every step, the agent selects a tactical skill from the flat tactical layer and executes it. Tactical skills grow (TD pre-filter → LLM judgment → immediate storage), shrink (utility-modulated decay → pruning), and are periodically abstracted (sleep consolidation → $d=1$ scaffold). The strategic layer grows only through sleep consolidation.

**Key data structures:**
- `SkillNode` — one per skill at any depth. Strategic (layer 1) and tactical (layer 2) nodes share the same class but populate different fields. Defined in §6.3.
- `SkillGraph` — backed by SQLite via SQLAlchemy (§6.1.1). Children derived via query on `parent_id`.
- `EpisodicMemoryBank` — separate store of raw experiences, linked from nodes via `evidence_ids`.

**Execution order per episode:**
1. Classify task type → `t_k`
2. **Strategic selection (once):** select $d=1$ scaffold via option-value retrieval (§9.1); null if $d=1$ is empty
3. For each step: tactical retrieval → execute → compute TD error → update Q → TD pre-filter check → LLM judgment if admitted → immediate node creation if approved
4. End of episode: update strategic option-value $Q^\Omega$ → graph maintenance (decay + pruning) → recompute `decay_rate` for active nodes → check sleep-consolidation trigger (§6.6)

**Critical invariants:**
- `decay_rate` on a tactical node always equals `λ / (Q_salience + ε)`. Recompute after every Q-update. Strategic ($d=1$) nodes always have `decay_rate = 0.0`.
- No `children_index`. `parent_id` is the single source of truth; children derived via SQL query.
- `total_accessed` is a `@property`, never stored.
- All new tactical nodes enter at depth `2` (flat tactical layer). No exceptions.
- Strategic Q-values (`Q_omega`) and tactical Q-values (`Q`) are **separate and must never be merged**.
- Bootstrap phase: $d=1$ is seeded manually or via LLM reflection until first sleep consolidation fires.
- Pilot scope: Q-learning operates over the tactical layer only until the first $d=1$ node exists.

---

## 1. Introduction and Motivation

Standard agent memory systems conflate several distinct questions:

- Which experience is worth storing?
- Which stored experience is worth keeping?
- Which kept experience generalizes to new tasks?

Most prior work (MemGPT, A-MEM, Voyager, SkillLib) optimizes retrieval — choosing what to surface at inference time — but treats memory formation and consolidation as secondary. MemRL, the base template for this work, uses TD error as a reactive write-gate signal and delegates all memory quality judgment to the backbone LLM. This works for large frontier models with strong meta-cognitive capacity, but conflates formation, retention, and abstraction into a single undifferentiated mechanism.

This work separates these three questions:

- **Formation** is gated by TD error (cheap, algorithmic) followed by LLM semantic judgment (expensive, high-quality).
- **Retention** is governed by utility-modulated Ebbinghaus decay (algorithmic, continuous).
- **Abstraction** is handled by periodic sleep consolidation with LLM synthesis (batch, principled).

The central hypothesis is that the LLM's strength is in semantic judgment and abstraction — not in deciding how often to retrieve, how long to retain, or when to consolidate. Offloading those structural decisions to an algorithmic layer produces a more principled and debuggable memory system.

**Key design decisions confirmed for Phase 1:**

- Two-tier graph: $d=1$ strategic scaffolds (options, once per episode) and a flat tactical layer (skills, every step).
- Tactical formation: TD error pre-filter → LLM judges worth → immediate storage. No accumulation pool. No hard utility threshold. Decay removes what the LLM misjudged.
- Retention: Ebbinghaus decay modulated by confidence-weighted mean utility $\bar{Q}_{i,w}$ across task types. Both tactical and strategic Q-values stored per task type.
- Sleep consolidation: sole $d=1$ population mechanism. Periodic batch clustering of surviving tactical memories above a utility eligibility threshold. LLM judges generalizability of each cluster. Absorb-or-spawn decision determines whether a cluster extends an existing scaffold or creates a new one.
- Strategic scaffolds never decay. They are permanent in Phase 1.
- $Q^\Omega$ is per-task-type. Initialization for spawned scaffolds: shrinkage-weighted mean over absorbed cluster members' per-task-type Q-values — not zero. See §3.5.

**Explicitly deferred to Phase 2:**
- Transferability scoring ($\hat{T}$), float-up mechanism, depth differentiation within tactical layer
- Affect/personalization graph
- Learned formation policy $\pi_{\text{form}}$ replacing TD pre-filter
- DAG extension for multi-parent nodes
- Memory-quality bonus term in reward
- Double Q-learning for overestimation bias correction

---

## 2. Problem Formulation

### 2.1 MDP Definition

$$\mathcal{MDP} = \left(S,\ A^{\Omega},\ A^{\tau},\ P,\ R,\ \gamma,\ \mathcal{M}\right)$$

The memory bank $\mathcal{M}$ is a **side-channel** conditioning the policy, not part of the state space $S$. The action space is partitioned into $A^{\Omega}$ (strategic options, $d=1$) and $A^{\tau}$ (tactical actions, flat layer). This is a semi-MDP over $A^{\Omega}$ nested around a standard MDP over $A^{\tau}$, following **Sutton, Precup & Singh's Options framework (1999)**. Embedding $\mathcal{M}$ in $S$ would make the state space grow with every new skill, breaking convergence guarantees.

### 2.2 State

$$s_t = \left(t_k,\ c_t,\ h_t,\ \omega\right)$$

| Component | Description |
|---|---|
| $t_k$ | Task type. Fixed within an episode. |
| $c_t$ | Task context at step $t$. |
| $h_t$ | Short-term interaction history over last $w$ steps. |
| $\omega$ | Active strategic scaffold. Selected once at $t=0$, fixed for the episode. |

### 2.3 Action

$$a_0^{\Omega} = \omega \in \mathcal{G}_{d=1} \qquad \text{once, at } t=0$$

$$a_t^{\tau} = s_i \in \mathcal{G}_{\tau} \qquad \text{every step } t \geq 0$$

$\omega$ conditions reasoning context — it does not produce an environment transition directly. $a_t^\tau$ is passed to `env.step`. Token-level generation is outside this MDP.

### 2.4 Transition

$$s_{t+1} = \left(t_k,\ c_{t+1},\ h_{t+1},\ \omega\right)$$

$\omega$ and $t_k$ are invariant within an episode.

### 2.5 Reward

$$r_t = r_t^{\text{env}}$$

Per-step environment feedback attributed to the active tactical action $a_t^\tau$. Memory-quality bonus deferred:

$$r_t^{\text{full}} = r_t^{\text{env}} + \beta \cdot r_t^{\text{mem}}, \qquad \beta = 0 \text{ in Phase 1}$$

### 2.6 Discount Factor

$$\gamma \in [0.9,\ 0.99], \qquad \gamma^\Omega \in [0.9,\ 0.99]$$

$\gamma$ governs tactical single-step TD updates. $\gamma^\Omega$ governs the strategic option-value update over the full episode return. These are **separate hyperparameters**, swept independently. Sharing a single $\gamma$ across both tiers is architecturally incorrect: tactical $\gamma$ controls step-to-step dependency; strategic $\gamma^\Omega$ controls episode-level return attribution. Conflating them (as in the Options framework's single-discount formulation) introduces systematic bias in $Q^\Omega$ estimates when episodes are long. **Reference:** Sutton, Precup & Singh (1999) use separate intra-option and semi-MDP discounts — Phase 1 follows this convention.

### 2.7 Memory Bank

$$\mathcal{M}_t = \left(\mathcal{G}_t,\ \{Q_i(t_k)\},\ \{Q^{\Omega}_j(t_k)\},\ \lambda,\ \epsilon\right)$$

| Component | Description |
|---|---|
| $\mathcal{G}_t$ | Unified skill graph: $d=1$ strategic nodes + flat tactical layer |
| $\{Q_i(t_k)\}$ | Tactical Q-estimates, per task type. Decay salience uses shrinkage-weighted mean $\bar{Q}_{i,w}$. |
| $\{Q^{\Omega}_j(t_k)\}$ | Strategic option-values, per task type. **Separate from** $\{Q_i(t_k)\}$ — never merged. |
| $\lambda$ | Base decay rate (single value — flat tactical layer, no depth-indexing) |
| $\epsilon$ | Utility floor for decay denominator |

$\mathcal{M}$ is updated **after each episode**. Tactical Q-updates happen per step but are committed at episode end.

---

## 3. Utility Estimation

### 3.1 Semantics

$$Q_i(t_k) \approx \mathbb{E}\!\left[\Delta R \mid s_i,\ t_k\right], \qquad \Delta R = R_{\text{with skill}} - R_{\text{without skill}}$$

Q-values are stored **per task type** for both tactical and strategic nodes. Per-type granularity is required for calibrated decay salience and is the foundation for transferability scoring in Phase 2.

### 3.2 Tactical Q-Learning Update

$$Q_i(t_k) \leftarrow Q_i(t_k) + \alpha \Bigl[r_t + \gamma \max_{s_j \in \mathcal{N}(s_i)} Q_j(t_k) - Q_i(t_k)\Bigr]$$

### 3.3 Decay Salience — Confidence-Weighted Mean

Decay is governed by a **task-agnostic** salience denominator — the shrinkage-weighted mean utility across all task types a skill has been retrieved on:

$$\bar{Q}_{i,w} = \frac{\sum_k w_{ik} \cdot Q_i(t_k)}{\sum_k w_{ik}}, \qquad w_{ik} = \frac{n_{ik}}{n_{ik} + \lambda_{\text{shrink}}}$$

**Cold-start:** for a node observed only on $t_{k_0}$, shrinkage weights cancel and $\bar{Q}_{i,w} = Q_i(t_{k_0})$. Well-defined from first retrieval.

**Why not task-local $Q_i(t_k)$:** decay governs global graph membership in a unified (non-partitioned) graph. Using a task-local value makes retention path-dependent on whichever task type happened to run last — architecturally incoherent. $\bar{Q}_{i,w}$ is task-agnostic and reflects the skill's aggregate contribution across all contexts.

### 3.4 TD Error

$$\delta_t = r_t + \gamma \max_{s_j \in \mathcal{N}(s_i)} Q_j(t_k) - Q_i(t_k)$$

Drives the Q-update and serves as the Stage 1 pre-filter signal for tactical formation (§4.1).

### 3.5 Initialization

**Tactical nodes:** $Q_i(t_k) = 0\ \forall t_k$. No utility prior at creation — decay removes misjudged nodes quickly via the maximum decay rate $\lambda / \epsilon$ when Q-salience is zero.

**Strategic nodes — spawn case** (new $d=1$ node created by consolidation):

$$Q^{\Omega}_\omega(t_k) = \frac{\sum_{j \in \text{cluster}} w_j \cdot Q_j(t_k)}{\sum_{j \in \text{cluster}} w_j}, \qquad w_j = \frac{n_{jk}}{n_{jk} + \lambda_{\text{shrink}}}$$

Shrinkage-weighted mean over the absorbed cluster's per-task-type Q-values. Nodes with more evidence on $t_k$ contribute more. Task types not observed by any cluster member are absent from $Q^\Omega$ at creation — cold task type fallback (§9.1) handles this at retrieval time by normalizing the scaffold's per-task counts into a task distribution and taking the expected value under that distribution.

**Do not initialize spawned scaffolds to zero** — this causes systematic under-selection post-consolidation (FeUdal Networks dead-layer problem, Vezhnevets et al., 2017): zero-initialized scaffolds are never selected, so $Q^\Omega$ never gets updated, so they remain at zero indefinitely.

**Strategic nodes — absorb case** (existing tactical node promoted to $d=1$): carries tactical Q-values forward with dampening:

$$Q^{\Omega}_\omega(t_k) \leftarrow \rho \cdot Q_\tau(t_k), \qquad \rho \in (0.5,\ 1.0)$$

$\rho$ corrects for the fact that Q-values calibrated in the tactical competitive context may overestimate strategic value when the node moves to a different competitive tier. Starting value: $\rho = 0.7$, swept in ablation.

### 3.6 Tactical Action Selection

$$a_t^{\tau} = \arg\max_{s_i \in \text{top-}k}\ Q_i(t_k)$$

Candidates shortlisted by semantic similarity, then ranked by per-task-type Q-value for the current episode's $t_k$. Retrieval technique open (§11).

### 3.7 Failure Credit Assignment

When $\delta_t < -\theta_\delta$, distribute penalty across active tactical skills using recency-weighted credit:

$$\Delta Q_s(t_k) = -|\delta_t| \cdot \gamma^{T - \text{step}(s)}$$

Skills most recently active receive the largest penalty. $\omega$ is excluded — its credit is resolved once at episode end (§3.8).

> **Known limitation:** recency weighting is causally imprecise for multi-step reasoning chains where the setup step (not the final step) is the actual failure point. Causal credit assignment via a learned model is a Phase 2 item.

> **Implementation note:** maintain `active_skills: list[tuple[SkillNode, int]]` — tactical nodes only — during each episode.

### 3.8 Strategic Option-Value Update

Updated once per episode at episode end:

$$Q^{\Omega}_{\omega}(t_k)\ \leftarrow\ Q^{\Omega}_{\omega}(t_k)\ +\ \alpha^{\Omega} \Bigl[\sum_{t=0}^{T-1} (\gamma^\Omega)^t r_t\ -\ Q^{\Omega}_{\omega}(t_k)\Bigr]$$

Uses $\gamma^\Omega$, not tactical $\gamma$. No per-step bootstrap term in Phase 1 (scaffold runs to episode termination, no early termination). **Storage:** `Q_omega` dict, never merged with tactical `Q`.

**Cold task type fallback:** if $Q^\Omega_{\omega_j}(t_k)$ is undefined for the current task type on every scaffold, score each scaffold by its normalized per-task distribution:

$$p_{\omega_j}(t_\ell) = \frac{n^\Omega_{\omega_j}(t_\ell)}{\sum_m n^\Omega_{\omega_j}(t_m)}$$

$$\bar{Q}^\Omega_{\omega_j} = \sum_{\ell} p_{\omega_j}(t_\ell)\, Q^\Omega_{\omega_j}(t_\ell)$$

Then select the scaffold with the highest $\bar{Q}^\Omega_{\omega_j}$. If a scaffold has no task counts yet, its fallback score is $0$. The issue this addresses is that a cold task has no direct option-value estimate, and a raw or unnormalized mean can unfairly favor scaffolds with sparse or skewed task histories.

---

## 4. Tactical Formation Pipeline

Raw experiences do not directly become skill nodes. A two-stage pipeline controls admission.

```
Raw experience (every step)
      ↓
  Stage 1: TD error pre-filter (cheap, algorithmic)
      ↓ (positive surprise only)
  Stage 2: LLM judgment (semantic, quality gate)
      ↓ (if approved)
  SkillNode created immediately (no accumulation)
  Decay handles pruning of misjudged nodes
```

### 4.1 Stage 1 — TD Error Pre-Filter

$$\delta_t > \theta_\delta \Rightarrow \text{pass to LLM judgment}$$

| Condition | Action |
|---|---|
| $\delta_t > \theta_\delta$ | Positive surprise — pass to Stage 2 |
| $\delta_t < -\theta_\delta$ | Negative surprise — apply failure credit (§3.7), discard for formation |
| $|\delta_t| \leq \theta_\delta$ | Expected outcome — discard entirely |

**Purpose:** eliminates the vast majority of experiences before any LLM call. TD error is already computed in the training loop at zero marginal cost. Only genuinely surprising positive experiences warrant LLM judgment.

**Why positive surprise only:** negative surprises provide Q-correction signal (§3.7) but do not represent skills worth forming. Avoidance skill formation (negative-experience nodes) is a Phase 2 item — acknowledged as a known gap in §11.

### 4.2 Stage 2 — LLM Judgment

The LLM receives the experience (state, action, reasoning trace, outcome) and judges:

1. **Is this experience semantically coherent** as a reusable skill?
2. **Is it distinct enough** from existing tactical memories (checked via embedding similarity against current graph)?
3. **Does it represent genuine capability**, not environmental stochasticity?

If all three pass → `SkillNode` created immediately at depth $\tau$ (flat tactical layer).

**No accumulation pool:** the old `CandidateRecord` / Gate 2 count threshold is removed. The LLM judgment replaces the evidence-accumulation pre-filter. Decay removes nodes that the LLM misjudged — a high-rate-of-decay node that is never retrieved will be pruned within $\theta_{\text{prune}}$ steps regardless of how confidently it was formed.

> **Implementation note:** LLM judgment calls are batched at end-of-episode, not inline during the step loop. Collect all Stage 1 admissions during the episode; run LLM judgment in batch; create nodes for approved experiences; commit all new nodes to `skill_graph_state` in one write.

> **Implementation note:** node creation requires LLM skill extraction to populate `content` and an embedding model call to populate `embedding`, written to `skill_representation` (§6.1.1) keyed by `node_id`. Both are I/O operations — batch at episode end.

---

## 5. Unified Skill Graph

### 5.1 Structure

$$\mathcal{G} = (V,\ E)$$

| Component | Description |
|---|---|
| $V$ | All `SkillNode` objects plus one virtual root $r$ |
| $E$ | Parent → child directed edges |
| Parent constraint | Each node has exactly one parent (tree, Phase 1) |
| Cross-edges | None in Phase 1 |

**Depth assignment:**

$$\text{depth}(s_i) = \begin{cases} 0 & \text{virtual root} \\ 1 & \text{strategic scaffold — sleep consolidation only} \\ \tau & \text{tactical skill — flat layer, all tactical nodes} \end{cases}$$

There is no depth differentiation within the tactical layer in Phase 1. All tactical nodes are peers. Float-up, transferability scoring, and intra-tactical depth are Phase 2 items.

No `children_index` is maintained. `parent_id` is the single source of truth; children derived via `SELECT node_id FROM skill_graph_state WHERE parent_id = ?`.

**Phase 2 extension:** `secondary_parents: list[str]` reserved, empty in Phase 1.

### 5.2 Bootstrap Phase

Until the first sleep consolidation event fires, $d=1$ is empty or manually seeded. The agent operates with $\omega = \text{null}$ (no strategic conditioning) and the Q-learning loop covers the tactical layer only. No formal gating on bootstrap-seeded $d=1$ nodes; they are subject to normal sleep-consolidation absorption logic once regular consolidation begins.

### 5.3 Storage Backend

**SQLite via SQLAlchemy**, consistent with MemRL's `MemoryService`. Two tables:

```sql
-- Write-once at creation. content and embedding never diverge.
CREATE TABLE skill_representation (
    node_id     TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   BLOB NOT NULL    -- numpy.ndarray.tobytes(); np.frombuffer() to deserialize
);

-- Mutable algorithmic state. Updated every episode.
CREATE TABLE skill_graph_state (
    node_id             TEXT PRIMARY KEY,
    depth               INTEGER NOT NULL,       -- 1 (strategic) or 2 (tactical)
    parent_id           TEXT,                   -- NULL only for virtual root
    task_type_dominant  TEXT,                   -- argmax_k n(t_k); updated dynamically
    t_create            INTEGER,
    last_accessed_step  INTEGER,
    decay_rate          REAL DEFAULT 0.0,
    consolidated        INTEGER DEFAULT 0,      -- boolean 0/1; layer 2 only
                                                -- True once this node has been processed
                                                -- by a sleep consolidation event
    Q                   TEXT,                   -- JSON dict[str, float]: per-task-type Q-values; tactical only
    n                   TEXT,                   -- JSON dict[str, int]: per-task-type retrieval counts; tactical only
    Q_omega             TEXT,                   -- JSON dict[str, float]: per-task-type option-values; strategic only
    n_omega             TEXT,                   -- JSON dict[str, int]: episode counts per task type; strategic only
    evidence_ids        TEXT                    -- JSON list[str], reservoir-capped at R
);

CREATE INDEX idx_parent ON skill_graph_state(parent_id);
CREATE INDEX idx_depth  ON skill_graph_state(depth);
CREATE INDEX idx_consolidated ON skill_graph_state(consolidated);  -- for sleep trigger query
```

**Working-set protocol:** load relevant `SkillNode` objects into in-memory working set at episode start. All step-level mutation happens in-memory. Flush to `skill_graph_state` in one batch write at episode end. SQLite is the durable store; the working set is scratch space for one episode.

**Embeddings computed once at creation**, never recomputed on read. Query embedding $e_q$ is the only embedding computed at inference time.

**`task_type_dominant`** is dynamic: $\arg\max_k n_{ik}$ from the retrieval count dict. Updated at episode end when `n` is flushed. Replaces the old static `task_type_primary` (formation-time artifact). For strategic nodes: dominant task type across the absorbed cluster at consolidation time.

### 5.4 Node Schema — Phase 1

```python
from dataclasses import dataclass, field

@dataclass
class SkillNode:
    # --- Identity ---
    id: str                          # UUID. Joins to skill_representation.node_id.

    # --- Provenance ---
    task_type_dominant: str          # argmax_k n(t_k). Dynamic — updated at episode end.
                                     # For strategic nodes: dominant task type of absorbed cluster.
    t_create: int                    # Global step at creation.

    # --- Hierarchy ---
    depth: int                       # 1 (strategic) or TAU (tactical flat layer).
    parent_id: str | None            # Single source of truth for tree structure.
                                     # Children derived via SQL query; never stored redundantly.
    secondary_parents: list[str] = field(default_factory=list)  # Phase 2. Empty in Phase 1.

    # --- Usage ---
    last_accessed_step: int = 0

    # --- Tactical Utility (layer 2 ONLY) ---
    Q: dict[str, float] = field(default_factory=dict)
                                     # Per-task-type Q-values: Q[t_k] = Q_i(t_k).
                                     # Empty/unused on strategic nodes.
    n: dict[str, int] = field(default_factory=dict)
                                     # Per-task-type retrieval counts: n[t_k] = n_ik.
                                     # Used for shrinkage weights in Q_bar_w and decay.
                                     # Empty/unused on strategic nodes.

    # --- Strategic Option-Value (layer 1 ONLY) ---
    Q_omega: dict[str, float] = field(default_factory=dict)
                                     # Per-task-type option-values: Q_omega[t_k].
                                     # Initialized from cluster shrinkage-weighted mean (§3.5).
                                     # SEPARATE from Q — never merge.
                                     # Empty/unused on tactical nodes.
    n_omega: dict[str, int] = field(default_factory=dict)
                                     # Episode count scaffold was selected, per task type.
                                     # Empty/unused on tactical nodes.

    # --- Retention ---
    decay_rate: float = 0.0          # Cached: λ / (Q_bar_w + ε). Always 0.0 for d=1.
                                     # Recomputed after every Q-update via recompute_decay_rate().

    # --- Episodic Links ---
    evidence_ids: list[str] = field(default_factory=list)
                                     # IDs into EpisodicMemoryBank. Reservoir-capped at R.

    # --- Sleep Consolidation Bookkeeping (layer 2 ONLY) ---
    consolidated: bool = False       # True once processed by any sleep consolidation event
                                     # (whether absorbed, spawned into, or judged non-general).
                                     # Drives the unconsolidated-count sleep trigger (§8.1).
                                     # Meaningless on d=1 nodes — always False there.

    # --- Derived ---
    @property
    def total_accessed(self) -> int:
        """Total tactical retrievals across all task types. @property — never store separately."""
        return sum(self.n.values())

    def recompute_decay_rate(self, lambda_base: float, epsilon: float,
                              lambda_shrink: float = 10) -> None:
        """
        Recompute and cache decay rate. TACTICAL NODES ONLY.

        decay_rate = λ / (Q̄_w + ε)

        where Q̄_w is the shrinkage-weighted mean across ALL task types in self.Q.
        This is the GLOBAL salience denominator — task-agnostic, consistent with the
        unified non-partitioned graph.

        d=1 nodes: unconditionally sets decay_rate = 0.0. Strategic nodes never decay.
        """
        if self.depth == 1:
            self.decay_rate = 0.0
            return
        q_bar_w = self._weighted_mean_utility(lambda_shrink)
        self.decay_rate = lambda_base / (q_bar_w + epsilon)

    def _weighted_mean_utility(self, lambda_shrink: float = 10) -> float:
        """
        Shrinkage-weighted mean over the per-task-type Q dict.
        Q̄_w = Σ w_ik * Q(t_k) / Σ w_ik,  w_ik = n_ik / (n_ik + λ_shrink)
        Returns 0.0 if Q is empty (new node, no retrievals yet → maximum decay rate).
        """
        if not self.Q:
            return 0.0
        weighted_sum, weight_sum = 0.0, 0.0
        for t_k, q in self.Q.items():
            n_ik = self.n.get(t_k, 0)
            w = n_ik / (n_ik + lambda_shrink)
            weighted_sum += w * q
            weight_sum += w
        return weighted_sum / weight_sum if weight_sum > 0.0 else 0.0
```

**Field notes:**

| Field | Notes |
|---|---|
| `id` | Primary key joining both tables. |
| `task_type_dominant` | Dynamic — updated at episode end from `argmax(n)`. Not static formation-time artifact. |
| `decay_rate` | Tactical: `λ / (Q̄_w + ε)`. Strategic: always `0.0`. Never compute retention without calling `recompute_decay_rate()` first. |
| `Q` vs `Q_omega` | Mutually exclusive by depth. Assert `depth == 1 ⟹ Q empty` and `depth == TAU ⟹ Q_omega empty`. Both are `dict[str, float]` keyed by task type. |
| `consolidated` | Layer 2 only. Drives sleep trigger counter. Do not repurpose. |
| `evidence_ids` | Reservoir-sampled at cap $R$. Implement `add_evidence(eid)` with reservoir sampling. |
| `total_accessed` | `@property` over `n`. Never store separately — it will diverge. |

**`content` and `embedding`** live in `skill_representation`, not on `SkillNode`. For tactical nodes: LLM-extracted procedural summary + embedding. For strategic nodes: LLM-synthesized abstraction from cluster contents + either cluster centroid embedding or fresh embedding of synthesized content (pick one, document it — do not leave ambiguous).

---

## 6. Memory Decay (Tactical Layer Only)

### 6.1 Design Rationale

Strategic ($d=1$) nodes are categorically permanent — not merely assigned zero decay rate as a special case. They are outside the decay/pruning mechanism entirely.

Tactical decay governs global graph membership. Using task-local $Q_i(t_k)$ as the salience denominator would make retention path-dependent on the last episode's task type, which is architecturally incoherent for a unified (non-partitioned) graph. The confirmed salience denominator is $\bar{Q}_{i,w}$ — the shrinkage-weighted mean across all task types the skill has been retrieved on. This is task-agnostic, consistent with the unified graph design, and well-defined from the first retrieval (§3.3).

### 6.2 Formula

$$d_i(\Delta t) = e^{-\text{decay\_rate} \cdot \Delta t}$$

$$\text{decay\_rate} = \frac{\lambda}{\text{Q\_salience} + \epsilon}$$

| Term | Description |
|---|---|
| $\lambda$ | Base decay rate (single value; no depth-indexing in Phase 1 flat tactical layer) |
| $\Delta t$ | Retrieval steps elapsed since `last_accessed_step` (not wall-clock) |
| $\bar{Q}_{i,w}$ | Shrinkage-weighted mean utility across task types (§3.3) |
| $\epsilon$ | Floor preventing division by zero. Starting value: $0.01$ |

### 6.3 Boundary Cases

| Condition | Effective rate | Consequence |
|---|---|---|
| $\bar{Q}_{i,w} = 0$ | $\lambda / \epsilon$ — maximum | New nodes with no confirmed utility are pruned quickly |
| $\bar{Q}_{i,w} \to 1$ | $\approx \lambda$ | High-utility nodes decay at base rate |
| $d=1$ | $0$ | Permanent |
| Single task type observed | $\bar{Q}_{i,w} = Q_i(t_{k_0})$ (shrinkage cancels) | Cold-start well-defined from first retrieval |

### 6.4 Recompute Protocol

After any Q-update to an active node:

```python
node.recompute_decay_rate(lambda_base=graph.lambda_base, epsilon=graph.epsilon,
                          lambda_shrink=graph.lambda_shrink)
```

Called on **active nodes only** — $O(\text{active})$, not $O(|G|)$.

---

## 7. Memory Management

Two complementary mechanisms control tactical layer size.

### 7.1 Decay-Based Pruning

$$d_i(\Delta t) < \theta_{\text{prune}} \Rightarrow \text{remove node}$$

Task-agnostic — uses `node.decay_rate` directly, no `t_k` dependency. Never prunes $d=1$ nodes.

```python
for node in list(G.tactical_nodes()):
    delta_t = current_step - node.last_accessed_step
    retention = exp(-node.decay_rate * delta_t)
    if retention < theta_prune:
        G.remove(node)
```

### 7.2 Tactical Action Space Cap

$$|A^\tau| \leq N$$

At retrieval time, only top-$N$ tactical nodes by Q-value are eligible. Hard computational guarantee on action space size, required for Q-learning convergence. Does not apply to strategic selection ($d=1$ population is small by construction).

| Mechanism | Type | Controls | Hyperparameter |
|---|---|---|---|
| Ebbinghaus decay + $\theta_{\text{prune}}$ | Soft, continuous | Graph membership | $\theta_{\text{prune}}$ |
| $\|A\| \leq N$ | Hard, discrete | Retrieval eligibility | $N$ |

---

## 8. Sleep Consolidation — Strategic Scaffold Formation

The **sole** mechanism by which $d=1$ nodes are created after the bootstrap phase. Periodic, batch, runs after graph maintenance.

### 8.1 Trigger Condition

Tracks **unconsolidated tactical nodes** — nodes not yet processed by any sleep event:

$$\text{count}_{\text{unconsolidated}} = |\{\, n \in \mathcal{G}_2 : \neg n.\texttt{consolidated} \,\}|$$

Sleep fires when:

$$\text{count}_{\text{unconsolidated}} \geq N_{\text{sleep}}$$

**Why unconsolidated count, not total tactical population:** total population fluctuates from decay-based pruning independent of consolidation. Gating on unconsolidated count means the trigger fires only in response to genuinely new, unprocessed material since the last sleep event. Pruning an unconsolidated node removes it from the counter without triggering a spurious sleep event; pruning a consolidated node has no counter effect.

**Consolidation eligibility filter (pre-LLM):** only tactical nodes with Q\_salience $> \theta_{\text{consolidate}}$ are passed to clustering. This prevents low-utility survivors (nodes in the process of decaying out) from polluting the consolidation input and avoids clustering thousands of nodes at each sleep event. $\theta_{\text{consolidate}}$ is a cheap arithmetic filter, not an LLM call.

### 8.2 Consolidation Procedure

```python
def sleep_consolidation(graph: SkillGraph, theta_absorb: float,
                        theta_consolidate: float) -> None:

    # Eligibility filter — cheap, pre-LLM
    eligible = [n for n in graph.tactical_nodes()
                if not n.consolidated
                and q_salience(n) > theta_consolidate]
    if not eligible:
        return

    # Step 1: cluster eligible nodes by embedding similarity
    embeddings = {n.id: graph.get_embedding(n.id) for n in eligible}
    clusters = cluster_embeddings(eligible, embeddings)  # HDBSCAN or k-means — open (§10)

    for cluster in clusters:
        centroid = mean_embedding([embeddings[n.id] for n in cluster])
        cluster_contents = [graph.get_content(n.id) for n in cluster]

        # Step 2: LLM judges generalizability of this cluster
        is_general = llm_judge_generalizability(cluster_contents)
        if not is_general:
            # Mark consolidated to prevent re-clustering; do not create d=1 node
            for node in cluster:
                node.consolidated = True
            continue

        # Step 3: check absorption against existing d=1 scaffolds
        existing_d1 = graph.nodes_at_depth(1)
        if existing_d1:
            d1_embeddings = {p.id: graph.get_embedding(p.id) for p in existing_d1}
            best_parent, similarity = max(
                ((p, cosine_sim(centroid, d1_embeddings[p.id])) for p in existing_d1),
                key=lambda x: x[1]
            )
        else:
            similarity = -1.0  # force spawn

        if similarity >= theta_absorb:
            # Absorb: reparent cluster under existing scaffold
            for node in cluster:
                graph.reparent(node, best_parent)
                node.consolidated = True
            # Q_omega of best_parent updated as running average (optional — document policy)

        else:
            # Spawn: synthesize new d=1 scaffold from cluster
            content = llm_synthesize_scaffold(cluster_contents)
            new_id = new_uuid()
            graph.write_representation(new_id, content, centroid)

            # Q_omega initialization: shrinkage-weighted mean over cluster (§3.5)
            q_omega_init = shrinkage_weighted_mean_Q(cluster, graph.lambda_shrink)

            new_scaffold = SkillNode(
                id=new_id,
                task_type_dominant=majority_task_type(cluster),
                t_create=graph.current_step,
                depth=1,
                parent_id=graph.root_id,
                Q_omega=q_omega_init,
                n_omega={},
                decay_rate=0.0,
            )
            graph.insert(new_scaffold, parent=graph.root_id)
            for node in cluster:
                graph.reparent(node, new_scaffold)
                node.consolidated = True
```

**Key additions vs. old spec:**

- `theta_consolidate` pre-filter applied before clustering (cheap, prevents low-utility node pollution)
- LLM generalizability judgment added per cluster before absorb/spawn decision
- `consolidated` flag replaces `absorbed_by_sleep` — broader semantics covering both absorb and spawn outcomes, and the case where LLM judges a cluster as not general enough
- $Q^\Omega$ initialized from cluster Q-values, not zero (§3.5)

**Ordering constraint:** sleep consolidation runs strictly after decay-based pruning in the same maintenance pass. Pruning writes graph removals; consolidation writes `parent_id` updates. Sequencing prune-first ensures consolidation never reparents a node that has simultaneously been marked for removal.

---

## 9. Retrieval

> ⚠️ **OPEN — NOT FINALIZED.** Confirmed constraints below. Exact tactical shortlist method (ANN vs. BM25 vs. hybrid) remains an open design decision.

Two separate procedures at two cadences. Never merged into a single top-$k$.

### 9.1 Strategic Retrieval (Once Per Episode, $d=1$ only)

$$\omega = \arg\max_{\omega_j \in \mathcal{G}_{d=1}} Q^{\Omega}_{\omega_j}(t_k)$$

Full scan over $d=1$ (small by construction). No embedding step — choice driven entirely by option-value evidence. If $d=1$ empty, $\omega = \text{null}$.

**Cold task type:** fall back to the scaffold with the highest task-distribution-normalized expected option-value $\bar{Q}^\Omega_{\omega_j}$ computed from its per-task $Q^\Omega$ and $n^\Omega$.

### 9.2 Tactical Retrieval (Every Step, flat tactical layer)

**Retrieval score:**

$$\text{score}(s_i,\ \Delta t) = d_i(\Delta t) \cdot \cos(e_i,\ e_q)$$

where $d_i(\Delta t) = e^{-\text{decay\_rate} \cdot \Delta t}$ and $\cos(e_i, e_q)$ is cosine similarity of node embedding to query embedding.

**Constraints:**
1. Flat scan over all tactical nodes (no depth traversal in Phase 1)
2. Top-$N$ cap respected (§7.2)
3. $\omega$ conditions retrieval via prompt/context — never via Q-blending

**Shortlist method:** embedding nearest-neighbor search, BM25, or hybrid — open (§10).

---

## 10. Episode Update Loop

```python
# G               — SkillGraph
# current_step    — global step counter
# active_skills   — list[(SkillNode, step_index)], tactical only
# episode_rewards — list[float]
# pending_formations — list[experience], Stage 1 admissions awaiting LLM judgment

for each episode:
    t_k = classify_task(episode)
    active_skills, episode_rewards, pending_formations = [], [], []

    # STRATEGIC SELECTION (once)
    omega = select_strategic_scaffold(G, t_k)  # §9.1; null during bootstrap

    for each step t:

        # TACTICAL RETRIEVAL
        candidates = recall_tactical(query=c_t, task_type=t_k, k=5, N_cap=N)
        a_t = argmax(candidates, key=lambda s: Q(s, t_k))
        active_skills.append((a_t, t))

        # EXECUTION
        r_t, s_next = env.step(a_t)
        episode_rewards.append(r_t)

        # TACTICAL Q UPDATE
        delta_t = r_t + gamma * max(Q(s_j, t_k) for s_j in N(a_t)) - Q(a_t, t_k)
        update_Q(a_t, t_k, delta_t, alpha)                   # updates a_t.Q[t_k]
        a_t.n[t_k] = a_t.n.get(t_k, 0) + 1
        a_t.last_accessed_step = current_step
        a_t.recompute_decay_rate(graph.lambda_base, graph.epsilon, graph.lambda_shrink)

        # STAGE 1 — TD PRE-FILTER
        if delta_t > theta_delta:
            pending_formations.append(experience(s_t, a_t, r_t, t_k))
        elif delta_t < -theta_delta:
            for (s, step_s) in active_skills:
                penalty = abs(delta_t) * (gamma ** (current_step - step_s))
                update_Q(s, t_k, -penalty, alpha=1.0)        # updates s.Q[t_k]
                s.recompute_decay_rate(graph.lambda_base, graph.epsilon, graph.lambda_shrink)

        current_step += 1

    # END OF EPISODE

    # STRATEGIC UPDATE (once)
    if omega is not None:
        episode_return = sum((gamma_omega**t) * r for t, r in enumerate(episode_rewards))
        omega.Q_omega[t_k] = omega.Q_omega.get(t_k, 0.0) + alpha_omega * (
            episode_return - omega.Q_omega.get(t_k, 0.0)
        )
        omega.n_omega[t_k] = omega.n_omega.get(t_k, 0) + 1

    # STAGE 2 — LLM JUDGMENT (batched)
    approved = llm_judge_formations(pending_formations, G)  # batch call; returns subset
    for exp in approved:
        new_node = create_skill_node(exp)    # LLM extraction + embedding; depth=2
        parent = G.root_id                   # all tactical nodes hang from the virtual root in Phase 1
        G.insert(new_node, parent)

    # GRAPH MAINTENANCE — tactical only, task-agnostic
    for node in list(G.tactical_nodes()):
        delta_t_node = current_step - node.last_accessed_step
        retention = exp(-node.decay_rate * delta_t_node)
        if retention < theta_prune:
            G.remove(node)

    # UPDATE task_type_dominant for active nodes
    for node in [s for s, _ in active_skills]:
        node.task_type_dominant = argmax(node.n)

    # SLEEP CONSOLIDATION TRIGGER
    unconsolidated_count = sum(1 for n in G.tactical_nodes() if not n.consolidated)
    if unconsolidated_count >= N_sleep:
        sleep_consolidation(G, theta_absorb, theta_consolidate)  # §8.2
```

---

## 11. Open Problems

| Item | Status | Notes |
|---|---|---|
| Q-value representation | **Open (Phase 1a vs 1b)** | Start with global scalar (1a); switch to per-task-type weighted mean (1b) after debugging |
| Tactical retrieval technique | **Open** | ANN (FAISS/ScaNN) vs. BM25 vs. hybrid |
| Content representation | **Open** | Raw trace vs. distilled procedural summary |
| Embedding strategy | **Open** | Frozen LLM encoder vs. fine-tuned |
| Clustering method (sleep consolidation) | **Open** | HDBSCAN vs. k-means; minimum cluster size, distance metric |
| Task type definition $t_k$ | **Open** | Benchmark-derived, clustered, or fixed taxonomy |
| Scaffold embedding on absorb | **Open** | Fixed at creation vs. running average update |
| LLM judgment prompt design | **Open** | Formation judgment (Stage 2) and consolidation judgment prompts |
| Avoidance skill formation | **Known gap** | Negative-surprise experiences currently discarded; avoidance nodes deferred to Phase 2 |
| Causal credit assignment | **Known gap** | Recency-weighted failure credit is causally imprecise for multi-step chains; learned credit model deferred to Phase 2 |
| Task-dynamic normalization of Q for transfer | **Known gap** | $\bar{Q}_{i,w}$ conflates task-dynamic dissimilarity with skill specificity; normalization deferred to Phase 2 |
| Learned formation policy $\pi_{\text{form}}$ | **Deferred Phase 2** | Replaces TD pre-filter with off-policy learned classifier |
| Transferability scoring + float-up | **Deferred Phase 2** | $\hat{T}$, Gates 3–4, depth differentiation within tactical layer |
| Affect/personalization graph | **Deferred Phase 2** | Volatile user-preference memory |
| Double Q-learning | **Deferred Phase 2** | Overestimation bias correction |
| Memory-quality reward bonus | **Deferred Phase 2** | $r_t^{\text{mem}} = Q_i(t_k) - \bar{Q}(t_k)$ |
| DAG extension | **Deferred Phase 2** | Multi-parent nodes |

---

## 12. Hyperparameter Summary

| Symbol | Role | Starting value | Status |
|---|---|---|---|
| $\theta_\delta$ | TD error pre-filter threshold (Stage 1) | — | sweep |
| $\lambda$ | Base decay rate (flat tactical layer) | — | sweep |
| $\lambda_{\text{shrink}}$ | Bayesian shrinkage pseudocount for $\bar{Q}_{i,w}$ and $Q^\Omega$ initialization | $10$ | sweep |
| $\epsilon$ | Utility floor in decay denominator | $0.01$ | sweep |
| $\theta_{\text{prune}}$ | Retention threshold for tactical node removal | — | sweep |
| $N$ | Hard tactical action space cap | — | sweep |
| $\alpha$ | Tactical TD learning rate | $0.1$ | sweep |
| $\alpha^{\Omega}$ | Strategic option-value learning rate | $0.1$ | sweep, independent of $\alpha$ |
| $\gamma$ | Tactical discount factor | $[0.9, 0.99]$ | sweep |
| $\gamma^{\Omega}$ | Strategic discount factor (separate from $\gamma$) | $[0.9, 0.99]$ | sweep, independent of $\gamma$ |
| $R$ | Evidence reservoir size per node | $50$ | sweep |
| $N_{\text{sleep}}$ | Unconsolidated tactical count triggering sleep | — | sweep |
| $\theta_{\text{consolidate}}$ | Minimum $\bar{Q}_{i,w}$ for consolidation eligibility | — | sweep |
| $\theta_{\text{absorb}}$ | Min cluster-centroid-to-scaffold cosine for absorption | — | sweep |
| $\rho$ | Q-dampening factor on absorbed node promotion | $0.7$ | sweep |

**Removed from Phase 1 (deferred to Phase 2):**
$\theta_1$, $\theta_2$, $\theta_{\text{CV}}$, $N_{\min}$, $\epsilon_{\text{hyst}}$, $M_{\text{wait}}$, $\lambda_{\text{slow}}$, $\lambda_{\text{fast}}$

---

## 13. Relationship to MemRL

| Aspect | MemRL | This Work (Phase 1) |
|---|---|---|
| Memory structure | Flat bank | Two-tier: $d=1$ strategic scaffolds + flat tactical layer |
| Storage backend | SQLite via SQLAlchemy (`MemoryService`) | Same; two tables: write-once `skill_representation`, mutable `skill_graph_state` |
| Skill formation | All experiences stored | TD pre-filter → LLM judgment → immediate storage |
| Formation signal | LLM judgment only | TD error (algorithmic, cheap) gates before LLM (semantic, expensive) |
| Retention | Recency / retrieval frequency | Ebbinghaus decay modulated by utility salience (Phase 1a: global scalar; Phase 1b: shrinkage-weighted mean) |
| Abstraction | None | Periodic sleep consolidation: cluster surviving tactical memories → LLM synthesis → $d=1$ scaffold |
| Action space | Flat, single-tier | Two-tier: strategic option (once per episode) + tactical action (every step) |
| Action space bound | Unbounded | Hard cap $\|A^\tau\| \leq N$ + soft decay pruning |
| Utility signal | LLM-assessed at retrieval | Q-learning TD updates per task type (tactical); option-value full-episode return per task type (strategic) |
| Decay salience | N/A | $\bar{Q}_{i,w}$ — shrinkage-weighted mean across task types; task-agnostic; consistent with unified graph |
| Strategic scaffolds | None | Permanent $d=1$ nodes; never decay; $Q^\Omega$ per task type; initialized from cluster shrinkage-weighted mean, not zero |
| LLM dependency | All memory decisions | Semantic judgment only (formation quality, consolidation synthesis); structural decisions are algorithmic |
