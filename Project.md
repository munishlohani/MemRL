# Agent Memory with Utility-Based Skill Consolidation
## Architecture Specification — Phase 1
**Working Paper | Summer 2026**

---

## Abstract

We propose a memory architecture for AI agents that organizes skills within a two-tier hierarchical graph. The **strategic tier** ($d=1$) holds reasoning scaffolds — abstract frames selected once per episode under an options/semi-MDP formalism, with option-values stored per task type. The **tactical tier** (flat) holds directly executable skills formed from experience and retained via utility-modulated Ebbinghaus decay. Tactical skills are admitted by an advantage pre-filter (Monte Carlo return-to-go vs. a per-task-type baseline) followed by LLM judgment, stored immediately, and pruned by decay. Periodically, a **sleep consolidation** event clusters surviving tactical memories and uses LLM judgment to abstract them into strategic scaffolds — the sole mechanism by which $d=1$ nodes are created. The system is framed as a two-tier extended semi-MDP. Both tactical and strategic Q-values are stored **per task type**. Memory retention follows a biologically-grounded Ebbinghaus decay formula modulated by the **confidence-weighted mean utility** $\bar{Q}_{i,w}$ across task types — a task-agnostic salience denominator consistent with the unified (non-partitioned) graph design.

**Base template:** MemRL. This architecture extends MemRL by: (1) replacing flat memory with a two-tier hierarchical graph whose structure is determined by utility evidence and LLM abstraction rather than recency alone; (2) introducing a gated tactical formation pipeline with LLM judgment; (3) a separate sleep-consolidation pipeline for strategic scaffold formation; (4) options-style credit assignment for strategic actions; and (5) utility-modulated decay salience that governs global graph membership.

**Key departure from MemRL:** MemRL delegates all memory quality judgment to the backbone LLM's in-context reasoning at retrieval time. This architecture offloads structural decisions — what to form, what to retain, when to consolidate — to an algorithmic layer (advantage / MC return-to-go, decay, clustering), while trusting the LLM for semantic judgment (formation quality, consolidation content synthesis). The combination reduces the burden on the LLM while preserving its strength in semantic abstraction.

> **Status:** Phase 1 architecture confirmed. Utility estimator: **Monte Carlo return-to-go** (no bootstrap), committed at episode end. Both tiers store a per-task-type **mean advantage** (return-to-go minus a per-task-type baseline); selection, decay salience, consolidation eligibility, and $Q^\Omega$ init all read advantage. Decay salience is the shrinkage-weighted mean advantage floored at zero. Strategic scaffolds carry an advantage against a strategic baseline (penalized when their episodes underperform). $Q^\Omega$ init scale is resolved by advantage space — horizon inflation retired (§3.5).

---

## Implementation Notes for Coding Agent

This section provides a compact map of every component. Read this before touching any other section.

**What the system is:** A two-tier skill memory graph sitting alongside an LLM-based agent. At episode start, the agent selects one $d=1$ strategic scaffold (an option, held fixed for the whole episode) that conditions reasoning context. At every step, the agent selects a tactical skill from the flat tactical layer and executes it. Tactical skills grow (advantage pre-filter → LLM judgment → storage at episode end), shrink (utility-modulated decay → pruning), and are periodically abstracted (sleep consolidation → $d=1$ scaffold). The strategic layer grows only through sleep consolidation.

**Key data structures:**
- `SkillNode` — one per skill at any depth. Strategic (layer 1) and tactical (layer 2) nodes share the same class but populate different fields. Defined in §6.3.
- `SkillGraph` — backed by SQLite via SQLAlchemy (§6.1.1). Children derived via query on `parent_id`.
- `EpisodicMemoryBank` — separate store of raw experiences, linked from nodes via `evidence_ids`.

**Execution order per episode:**
1. Classify task type → `t_k`
2. **Strategic selection (once):** select $d=1$ scaffold via option-value retrieval (§9.1); null if $d=1$ is empty
3. For each step: tactical retrieval → execute → buffer step. At episode end: compute MC return-to-go → MC utility update → advantage pre-filter → LLM judgment for admitted steps → node creation if approved
4. End of episode: update strategic option-value $Q^\Omega$ → graph maintenance (decay + pruning) → recompute `decay_rate` for active nodes → check sleep-consolidation trigger (§6.6)

**Critical invariants:**
- `decay_rate` on a tactical node always equals `λ / (max(Q̄_w, 0) + ε)` where `Q̄_w` is the shrinkage-weighted mean advantage. Recompute after every utility update. Strategic ($d=1$) nodes always have `decay_rate = 0.0`.
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

Most prior work (MemGPT, A-MEM, Voyager, SkillLib) optimizes retrieval — choosing what to surface at inference time — but treats memory formation and consolidation as secondary. MemRL, the base template for this work, updates memory utility with a Monte Carlo terminal-reward rule (its Eq. 8, $Q \leftarrow Q + \alpha(r - Q)$ — a one-step-to-terminal collapse of the general TD form) and delegates all memory quality judgment to the backbone LLM. This works for large frontier models with strong meta-cognitive capacity, but conflates formation, retention, and abstraction into a single undifferentiated mechanism.

This work separates these three questions:

- **Formation** is gated by advantage (MC return-to-go vs. per-task-type baseline; cheap, algorithmic) followed by LLM semantic judgment (expensive, high-quality).
- **Retention** is governed by utility-modulated Ebbinghaus decay (algorithmic, continuous).
- **Abstraction** is handled by periodic sleep consolidation with LLM synthesis (batch, principled).

The central hypothesis is that the LLM's strength is in semantic judgment and abstraction — not in deciding how often to retrieve, how long to retain, or when to consolidate. Offloading those structural decisions to an algorithmic layer produces a more principled and debuggable memory system.

**Key design decisions confirmed for Phase 1:**

- Two-tier graph: $d=1$ strategic scaffolds (options, once per episode) and a flat tactical layer (skills, every step).
- Tactical formation: advantage pre-filter → LLM judges worth → storage at episode end. No accumulation pool. No hard utility threshold. Decay removes what the LLM misjudged.
- Retention: Ebbinghaus decay modulated by confidence-weighted mean utility $\bar{Q}_{i,w}$ across task types. Both tactical and strategic Q-values stored per task type.
- Sleep consolidation: sole $d=1$ population mechanism. Periodic batch clustering of surviving tactical memories above a utility eligibility threshold. LLM judges generalizability of each cluster. Absorb-or-spawn decision determines whether a cluster extends an existing scaffold or creates a new one.
- Strategic scaffolds never decay. They are permanent in Phase 1.
- $Q^\Omega$ is per-task-type. Initialization for spawned scaffolds: shrinkage-weighted mean over absorbed cluster members' per-task-type Q-values — not zero. See §3.5.

**Explicitly deferred to Phase 2:**
- Transferability scoring ($\hat{T}$), float-up mechanism, depth differentiation within tactical layer
- Affect/personalization graph
- Learned formation policy $\pi_{\text{form}}$ replacing advantage pre-filter
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

$\gamma$ is the discount *within the tactical MC return-to-go* $G_t = \gamma^{(T-1)-t} R$ (§3.2) — it sets how strongly the terminal reward is attributed to earlier steps in the episode. $\gamma^\Omega$ governs the strategic option-value update over the full episode return. These are **separate hyperparameters**, swept independently. Sharing a single $\gamma$ across both tiers conflates two timescales: tactical $\gamma$ controls intra-episode terminal-reward attribution; strategic $\gamma^\Omega$ controls episode-level return attribution. Conflating them introduces systematic bias in $Q^\Omega$ estimates when episodes are long (§14). **Reference:** Sutton, Precup & Singh (1999) use separate intra-option and semi-MDP discounts — Phase 1 follows this convention.

### 2.7 Memory Bank

$$\mathcal{M}_t = \left(\mathcal{G}_t,\ \{Q_i(t_k)\},\ \{Q^{\Omega}_j(t_k)\},\ \{b(t_k)\},\ \{b^\Omega(t_k)\},\ \lambda,\ \epsilon\right)$$

| Component | Description |
|---|---|
| $\mathcal{G}_t$ | Unified skill graph: $d=1$ strategic nodes + flat tactical layer |
| $\{Q_i(t_k)\}$ | Tactical mean **advantage**, per task type. Decay salience uses shrinkage-weighted mean $\bar{Q}_{i,w}$ floored at zero. |
| $\{Q^{\Omega}_j(t_k)\}$ | Strategic mean **advantage** (option-value), per task type. **Separate from** $\{Q_i(t_k)\}$ — never merged. |
| $\{b(t_k)\},\ \{b^\Omega(t_k)\}$ | Per-task-type advantage baselines: running mean terminal reward $R$ (tactical) and discounted return $G^\Omega$ (strategic). |
| $\lambda$ | Base decay rate (single value — flat tactical layer, no depth-indexing) |
| $\epsilon$ | Salience floor for decay denominator |

$\mathcal{M}$ is updated **after each episode**. All utility updates are computed from the buffered trajectory at episode end (no per-step commits).

---

## 3. Utility Estimation

### 3.1 Semantics

Each node stores a per-task-type **advantage** — its mean return-to-go relative to the per-task-type baseline:

$$Q_i(t_k) \approx \mathbb{E}\!\left[A_i(t_k)\right], \qquad A_i(t_k) = G_t - b(t_k)$$

where $b(t_k)$ is the running mean episode return for task type $t_k$ (§4.1). The stored value is an advantage, not a raw return: a skill scores positive only if episodes in which it was used beat the average outcome for that task type. This normalizes for task difficulty — a skill is not rewarded merely for appearing in easy episodes — and makes below-average skills negative, which the decay salience (§3.3) reads directly. Stored per task type for both tactical and strategic nodes.

> The field is named `Q` for schema continuity, but throughout Phase 1 it holds a **mean advantage**, not a Q-value. Treat "utility," "Q," and "mean advantage" as the same stored quantity.

### 3.2 Tactical Utility Update — Monte Carlo Return-to-Go

Committed once per episode, at episode end, for every tactical node retrieved during the episode. The update target is the realized **advantage** $A_t = G_t - b(t_k)$, where $G_t$ is the MC return-to-go and $b(t_k)$ the per-task-type baseline (§4.1). No bootstrap term.

$$Q_i(t_k) \leftarrow Q_i(t_k) + \alpha \bigl[A_t - Q_i(t_k)\bigr], \qquad A_t = G_t - b(t_k), \qquad G_t = \gamma^{(T-1)-t} R$$

Since intermediate rewards are zero and the only nonzero reward is the terminal outcome $R = r_{T-1}$ (§2.5), $G_t$ collapses to $\gamma^{(T-1)-t} R$. There is no `max` over graph neighbors — abstraction edges are not environment-transition edges, so a neighbor-max bootstrap has no MDP semantics (base MemRL collapses this to a terminal-state one-step update, its Eq. 8; here we keep the full discounted return-to-go so intermediate steps receive graded, recency-discounted credit). $\gamma$ is the discount *within the return*, not a bootstrap discount. Read the baseline before updating it (§4.1) so an episode is scored against history excluding itself.

This makes the tactical update consistent with the strategic update (§3.8), which was already bootstrap-free.

### 3.3 Decay Salience — Confidence-Weighted Mean

Decay is governed by a **task-agnostic** salience denominator — the shrinkage-weighted mean advantage across all task types a skill has been used on, floored at zero:

$$\bar{Q}_{i,w} = \frac{\sum_k w_{ik} \cdot Q_i(t_k)}{\sum_k w_{ik}}, \qquad w_{ik} = \frac{n_{ik}}{n_{ik} + \lambda_{\text{shrink}}}, \qquad \text{salience} = \max(\bar{Q}_{i,w},\ 0)$$

**Zero-floor is required, and it is also the correct behavior.** Since $Q_i(t_k)$ is now an advantage centered near zero, roughly half of nodes have $\bar{Q}_{i,w} < 0$; feeding a negative value into $\lambda/(\text{salience}+\epsilon)$ would give a negative or exploding decay rate. Flooring at zero maps any **below-baseline** skill to the maximum decay rate $\lambda/\epsilon$ — exactly what we want: a skill that performs worse than the task-type average should be pruned fast. Above-baseline skills ($\bar{Q}_{i,w}>0$) decay slower in proportion to their advantage.

**Cold-start:** for a node used only on $t_{k_0}$, shrinkage weights cancel and $\bar{Q}_{i,w} = Q_i(t_{k_0})$. Well-defined from first update.

**Why not task-local $Q_i(t_k)$:** decay governs global graph membership in a unified (non-partitioned) graph. Using a task-local value makes retention path-dependent on whichever task type ran last. $\bar{Q}_{i,w}$ is task-agnostic and reflects the skill's aggregate advantage across all contexts.

### 3.4 Return-to-Go and Advantage

No TD error is computed. The two derived signals are:

$$G_t = \gamma^{(T-1)-t} R \quad(\text{MC return-to-go, §3.2}) \qquad A_t = G_t - b(t_k) \quad(\text{advantage vs per-task-type baseline, §4.1})$$

$G_t$ is the update target for tactical utility (§3.2); $A_t$ is the Stage-1 formation gate signal (§4.1). Both are *computed* from the buffered trajectory at episode end — no reward model, no bootstrap.

### 3.5 Initialization

**Tactical nodes:** `Q` empty at creation (no task types seen yet). The first end-of-episode update writes the first advantage. Until then, salience is zero → maximum decay rate $\lambda/\epsilon$, so a node the LLM misjudged is pruned quickly if never used.

**Strategic nodes — spawn case** (new $d=1$ node created by consolidation). Initialize the scaffold's advantage per task type from the shrinkage-weighted mean advantage of its cluster members — **no horizon factor**:

$$Q^{\Omega}_\omega(t_k) = \frac{\sum_{j \in \text{cluster}} w_j \cdot Q_j(t_k)}{\sum_{j \in \text{cluster}} w_j}, \qquad w_j = \frac{n_{jk}}{n_{jk} + \lambda_{\text{shrink}}}$$

**Why no horizon normalization anymore.** Both tiers now store *advantage* — a difficulty-normalized quantity centered near zero (tactical: $G_t - b(t_k)$; strategic: $G^\Omega - b^\Omega(t_k)$, §3.8). They are on the same scale by construction, so the old $\frac{1}{1-\gamma^\Omega}$ inflation — which existed only to lift a per-step return estimate onto the episode-return scale — is unnecessary and would now systematically over-value spawned scaffolds. Dropping it also retires the `q_omega_init_horizon` mode and the W3 empirical-horizon apparatus (superseded: advantage space removes the scale mismatch they were correcting for). A spawned scaffold from a strong cluster inherits a positive advantage, so it is selected and updated — the FeUdal dead-layer failure (Vezhnevets et al., 2017) is avoided without inflation.

**Task types not observed** by any cluster member are absent from $Q^\Omega$ at creation — cold-task-type fallback (§9.1) handles this at retrieval time.

### 3.6 Tactical Action Selection

$$a_t^{\tau} = \arg\max_{s_i \in \text{children}(\omega)}\ Q_i(t_k) \qquad (\text{ties broken by } \bar{Q}_{i,w})$$

`Q` holds mean advantage (§3.1), so this ranks children of the active scaffold by task-difficulty-normalized utility for the current $t_k$. A child never yet used on $t_k$ has no `Q[t_k]` entry; treat its score as $\bar{Q}_{i,w}$ (its cross-task advantage) so it remains selectable rather than being locked out.

### 3.7 Failure Handling

No separate failure-credit mechanism. Failure is handled by the advantage update itself (§3.2): a failed episode yields low or negative $R$, so every retrieved node gets a low return-to-go $G_t = \gamma^{(T-1)-t} R$, hence a low (often negative) advantage $A_t = G_t - b(t_k)$, and its stored utility is pulled down by $Q_i(t_k) \leftarrow Q_i(t_k) + \alpha[A_t - Q_i(t_k)]$. A node that keeps landing below its task-type baseline drifts negative, its salience floors to zero, and it decays out. The $\gamma^{(T-1)-t}$ factor already supplies recency-graded credit — steps nearer termination absorb more of the terminal signal — so the old explicit $-|\delta_t|\cdot\gamma^{T-\text{step}}$ penalty is redundant and removed.

> **Known limitation:** MC return-to-go is causally imprecise — it credits/penalizes every step of an episode uniformly up to the $\gamma^{T-t}$ discount, not by actual causal contribution. Causal credit assignment via a learned model / PRM is a Phase 2 item (§11).

> **Implementation note:** maintain `active_skills: list[tuple[SkillNode, int]]` — tactical nodes only — during each episode, for the end-of-episode MC utility update.

### 3.8 Strategic Option-Value Update

Updated once per episode at episode end. The scaffold conditions the whole trajectory, so it is credited by the episode's **strategic advantage** — the discounted episode return minus a per-task-type strategic baseline. A scaffold whose episodes underperform the baseline accrues negative advantage and is deranked in selection (§9.1):

$$G^\Omega = \sum_{t=0}^{T-1} (\gamma^\Omega)^t r_t, \qquad A^\Omega = G^\Omega - b^\Omega(t_k), \qquad Q^{\Omega}_{\omega}(t_k)\ \leftarrow\ Q^{\Omega}_{\omega}(t_k)\ +\ \alpha^{\Omega}\bigl[A^\Omega - Q^{\Omega}_{\omega}(t_k)\bigr]$$

where $b^\Omega(t_k)$ is the running mean of $G^\Omega$ over episodes run under any scaffold on task type $t_k$ (read before update, like the tactical baseline). Uses $\gamma^\Omega$, not tactical $\gamma$. No per-step bootstrap (scaffold runs to termination). **Storage:** `Q_omega` dict, never merged with tactical `Q`.

**Cross-task summary (weighted mean).** The scaffold's task-agnostic value is the shrinkage-weighted mean advantage across the task types it has been selected on:

$$\bar{Q}^\Omega_{\omega} = \frac{\sum_k w^\Omega_{\omega k} \cdot Q^\Omega_\omega(t_k)}{\sum_k w^\Omega_{\omega k}}, \qquad w^\Omega_{\omega k} = \frac{n^\Omega_{\omega k}}{n^\Omega_{\omega k} + \lambda_{\text{shrink}}}$$

used for the cold-task-type fallback in §9.1. Weighting by selection count $n^\Omega_{\omega k}$ means a scaffold's summary reflects the task types it has actually been used on, in proportion to that evidence.

---

## 4. Tactical Formation Pipeline

Raw experiences do not directly become skill nodes. A two-stage pipeline controls admission. Both stages run **at episode end** over the buffered trajectory — not inline per step — because the admission signal (Monte Carlo return-to-go) is only defined once the terminal reward is known.

```
Buffered trajectory (whole episode)
      ↓
  Compute return-to-go G_t per step (arithmetic; no model)
      ↓
  Stage 1: advantage pre-filter  A_t = G_t − b(t_k) > θ_adv   (cheap, coarse episode-level gate)
      ↓ (above-baseline steps only)
  Stage 2: LLM judgment — intra-trajectory skill localization + quality gate
      ↓ (if approved)
  SkillNode created (no accumulation)
  Decay handles pruning of misjudged nodes
```

### 4.1 Stage 1 — Advantage Pre-Filter (Batched, End-of-Episode)

Intermediate rewards are zero; the only nonzero reward is the terminal task outcome $R$ (§2.5). The trajectory is buffered over the episode. At episode end we **compute** — not estimate; there is no reward model — the Monte Carlo return-to-go for each step:

$$G_t = \sum_{k \geq t} \gamma^{k-t} r_k = \gamma^{(T-1)-t} R \qquad (\text{since } r_k = 0 \text{ for } k < T-1)$$

Admission is gated on **advantage against a per-task-type baseline**, not raw return:

$$A_t = G_t - b(t_k) > \theta_{\text{adv}} \;\Rightarrow\; \text{pass step to Stage 2}$$

where $b(t_k)$ is the running mean episode return for task type $t_k$, tracked incrementally (Welford/EMA) — bookkeeping, not a model.

**What this gate does and does not do.** Under sparse terminal reward, $G_t$ has the *same sign for every step in an episode* — it is a coarse **episode-success** signal, not a per-step skill-quality signal. Subtracting $b(t_k)$ sharpens it to "this trajectory beat the average outcome for this task type," discarding mediocre episodes cheaply before any LLM call. It **cannot** isolate the load-bearing step within a successful trajectory; that intra-trajectory localization is delegated entirely to Stage 2.

**Division of labor (explicit):** Stage 1 is a cheap arithmetic *episode-level* gate (advantage sign). Stage 2 (§4.2) is LLM *intra-trajectory* localization + quality judgment, receiving every above-baseline step of an admitted episode. This is a deliberate reallocation: the RL signal is too coarse under sparse terminal reward to perform step-level credit assignment, so the judger absorbs that burden — at higher token cost, since a successful $T$-step episode yields up to $T$ candidates rather than the one or two a true surprise gate would emit.

**Known limitation (Phase 2):** distinguishing the causally-responsible step from incidental steps in a successful trajectory requires a per-step reward signal — a learned credit model or process reward model (Lightman et al. 2023) — which Phase 1 deliberately omits. See §11. Negative-outcome (avoidance) skill formation remains a Phase 2 item; episodes with $A_t \leq \theta_{\text{adv}}$ contribute no tactical formations.

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

Until the first sleep consolidation event fires, $d=1$ is empty or manually seeded. The agent operates with $\omega = \text{null}$ (no strategic conditioning) and the utility-update loop covers the tactical layer only. No formal gating on bootstrap-seeded $d=1$ nodes; they are subject to normal sleep-consolidation absorption logic once regular consolidation begins.

### 5.3 Storage Backend

**SQLite via SQLAlchemy**, consistent with MemRL's `MemoryService`. Two tables:

```sql
-- Write-once at creation. content and embedding never diverge.
CREATE TABLE skill_representation (
    node_id     TEXT PRIMARY KEY,
    content     TEXT NOT NULL,      -- LLM-generated summary. NOT raw experience trace.
                                    -- Kept concise for context-window efficiency at retrieval.
                                    -- Tactical: LLM-distilled procedural summary of the experience.
                                    -- Strategic: LLM-synthesized abstraction from cluster summaries.
    embedding   BLOB NOT NULL       -- Vector of content summary. numpy.ndarray.tobytes();
                                    -- np.frombuffer() to deserialize.
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

**Content generation:** at tactical node creation, the LLM is called once to produce a concise procedural summary of the experience — not the raw trace. The raw trace is stored in `EpisodicMemoryBank` via `evidence_ids` and is available for inspection but never surfaced directly at retrieval. This keeps retrieved content short enough to fit within the agent's context window when multiple nodes are retrieved per step.

**Embeddings computed once at creation**, over the LLM-generated summary, never recomputed on read. Query embedding $e_q$ is the only embedding computed at inference time.

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

            decay_rate = lambda_base / (max(Q_bar_w, 0) + epsilon)

        self.Q holds a per-task-type MEAN ADVANTAGE (may be negative). Q_bar_w is the
        shrinkage-weighted mean advantage across all task types in self.Q. It is FLOORED
        AT ZERO before use: a below-baseline skill (Q_bar_w < 0) gets the maximum decay
        rate lambda_base / epsilon and is pruned fast. Above-baseline skills decay slower.

        d=1 nodes: unconditionally decay_rate = 0.0 (strategic nodes never decay).
        """
        if self.depth == 1:
            self.decay_rate = 0.0
            return
        q_bar_w = self._weighted_mean_utility(lambda_shrink)
        salience = max(q_bar_w, 0.0)                      # advantage floor — see docstring
        self.decay_rate = lambda_base / (salience + epsilon)

    def _weighted_mean_utility(self, lambda_shrink: float = 10) -> float:
        """
        Shrinkage-weighted mean advantage over the per-task-type Q dict.
            Q_bar_w = sum(w_ik * Q[t_k]) / sum(w_ik),  w_ik = n_ik / (n_ik + lambda_shrink)
        May be negative (Q holds advantage). Returns 0.0 if Q is empty (new node → max decay).
        Caller floors at zero before using it as a decay denominator.
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
| `decay_rate` | Tactical: `λ / (max(Q̄_w,0) + ε)`. Strategic: always `0.0`. Never compute retention without calling `recompute_decay_rate()` first. |
| `Q` vs `Q_omega` | Mutually exclusive by depth. Both hold **mean advantage** (§3.1/§3.8), `dict[str, float]` keyed by task type, values may be negative. Assert `depth == 1 ⟹ Q empty` and `depth == TAU ⟹ Q_omega empty`. |
| `consolidated` | Layer 2 only. Drives sleep trigger counter. Do not repurpose. |
| `evidence_ids` | Reservoir-sampled at cap $R$. Implement `add_evidence(eid)` with reservoir sampling. |
| `total_accessed` | `@property` over `n`. Never store separately — it will diverge. |

**`content` and `embedding`** live in `skill_representation`, not on `SkillNode`. For tactical nodes: LLM-generated concise procedural summary + embedding of that summary. Raw experience trace is stored separately in `EpisodicMemoryBank` via `evidence_ids` — never surfaced directly at retrieval. For strategic nodes: LLM-synthesized abstraction from cluster summaries + embedding of that abstraction. Cluster centroid embedding is used as the initial embedding; can be replaced with a fresh embedding of the synthesized content — pick one, document it, do not leave ambiguous.

---

## 6. Memory Decay (Tactical Layer Only)

### 6.1 Design Rationale

Strategic ($d=1$) nodes are categorically permanent — not merely assigned zero decay rate as a special case. They are outside the decay/pruning mechanism entirely.

Tactical decay governs global graph membership. Using task-local $Q_i(t_k)$ as the salience denominator would make retention path-dependent on the last episode's task type, which is architecturally incoherent for a unified (non-partitioned) graph. The confirmed salience denominator is $\bar{Q}_{i,w}$ — the shrinkage-weighted mean across all task types the skill has been retrieved on. This is task-agnostic, consistent with the unified graph design, and well-defined from the first retrieval (§3.3).

### 6.2 Formula

$$d_i(\Delta t) = e^{-\text{decay\_rate} \cdot \Delta t}$$

$$\text{decay\_rate} = \frac{\lambda}{\max(\bar{Q}_{i,w},\ 0) + \epsilon}$$

| Term | Description |
|---|---|
| $\lambda$ | Base decay rate (single value; no depth-indexing in Phase 1 flat tactical layer) |
| $\Delta t$ | Retrieval steps elapsed since `last_accessed_step` (not wall-clock) |
| $\bar{Q}_{i,w}$ | Shrinkage-weighted mean **advantage** across task types (§3.3); floored at $0$ in the denominator |
| $\epsilon$ | Floor preventing division by zero. Starting value: $0.01$ |

### 6.3 Boundary Cases

| Condition | Effective rate | Consequence |
|---|---|---|
| $\bar{Q}_{i,w} \leq 0$ (below task-type baseline, or new node) | $\lambda / \epsilon$ — maximum | Below-average and unproven skills are pruned quickly |
| $\bar{Q}_{i,w}$ large positive | $\to$ small | Strongly-above-baseline skills are retained |
| $d=1$ | $0$ | Permanent |
| Single task type observed | $\bar{Q}_{i,w} = Q_i(t_{k_0})$ (shrinkage cancels) | Cold-start well-defined from first update |

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
    elapsed = current_step - node.last_accessed_step
    retention = exp(-node.decay_rate * elapsed)
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

**Consolidation eligibility filter (pre-LLM):** only tactical nodes whose salience $\max(\bar{Q}_{i,w}, 0) > \theta_{\text{consolidate}}$ are passed to clustering. Since $\bar{Q}_{i,w}$ is a mean advantage, this admits only skills that beat their task-type baseline by margin $\theta_{\text{consolidate}}$ — below-average survivors (still decaying out) are excluded before any LLM call. Cheap arithmetic filter, not an LLM call.

### 8.2 Consolidation Procedure

```python
def salience(node) -> float:
    """max(shrinkage-weighted mean advantage, 0). Same value used for decay (§3.3)."""
    return max(node._weighted_mean_utility(LAMBDA_SHRINK), 0.0)


def shrinkage_weighted_cluster_advantage(cluster) -> dict[str, float]:
    """
    Per-task-type shrinkage-weighted mean of the cluster's stored advantages.
    Returns {t_k: A} for every task type seen by any cluster member.
    NO horizon factor: tactical and strategic values are both advantages (§3.5).
        A[t_k] = sum(w_j * node.Q[t_k]) / sum(w_j),  w_j = n_jk / (n_jk + LAMBDA_SHRINK)
    """
    out, weights = {}, {}
    for node in cluster:
        for t_k, adv in node.Q.items():
            n_jk = node.n.get(t_k, 0)
            w = n_jk / (n_jk + LAMBDA_SHRINK)
            out[t_k] = out.get(t_k, 0.0) + w * adv
            weights[t_k] = weights.get(t_k, 0.0) + w
    return {t_k: out[t_k] / weights[t_k] for t_k in out if weights[t_k] > 0.0}


def sleep_consolidation(graph, theta_consolidate: float) -> None:
    # ---- Eligibility filter (cheap, pre-LLM): unconsolidated AND above-baseline ----
    eligible = [n for n in graph.tactical_nodes()
                if not n.consolidated and salience(n) > theta_consolidate]
    if not eligible:
        return

    # ---- Step 1: cluster eligible nodes by embedding similarity ----
    embeddings = {n.id: graph.get_embedding(n.id) for n in eligible}
    clusters = cluster_embeddings(eligible, embeddings)   # K-means; k selection open (§11)

    # ---- Step 2: one LLM decision per cluster ----
    for cluster in clusters:
        centroid = mean_embedding([embeddings[n.id] for n in cluster])
        cluster_contents = [graph.get_content(n.id) for n in cluster]
        existing_d1 = {p.id: graph.get_content(p.id) for p in graph.nodes_at_depth(1)}

        # LLM input : cluster_contents + existing_d1 (scaffold summaries)
        # LLM output: strict JSON, one object:
        #   {"action": "spawn" | "absorb" | "discard",
        #    "summary": str | null,               # required iff action == "spawn"
        #    "target_scaffold_id": str | null}     # required iff action == "absorb"
        decision = llm_decide_consolidation(cluster_contents, existing_d1)
        action = decision["action"]

        if action == "absorb":
            target = graph.get_node(decision["target_scaffold_id"])
            for node in cluster:
                graph.reparent(node, target)
                node.consolidated = True

        elif action == "discard":
            for node in cluster:
                node.consolidated = True          # marked, but no d=1 node created

        elif action == "spawn":
            new_id = new_uuid()
            graph.write_representation(new_id, decision["summary"], centroid)
            new_scaffold = SkillNode(
                id=new_id,
                task_type_dominant=majority_task_type(cluster),
                t_create=graph.current_step,
                depth=1,
                parent_id=graph.root_id,
                Q_omega=shrinkage_weighted_cluster_advantage(cluster),  # advantage, no horizon (§3.5)
                n_omega={},
                decay_rate=0.0,                    # strategic nodes never decay
            )
            graph.insert(new_scaffold, parent=graph.root_id)
            for node in cluster:
                graph.reparent(node, new_scaffold)
                node.consolidated = True

        else:
            raise ValueError(f"Unknown consolidation action: {action!r}")
```

**Key design decisions in this procedure:**

- `theta_consolidate` pre-filter applied before clustering (cheap arithmetic gate, prevents low-utility node pollution before LLM calls)
- LLM makes the absorb/spawn/discard decision in a single call — receives cluster contents and existing $d=1$ scaffold summaries; no cosine-threshold absorb gate
- `consolidated` flag covers all three outcomes (absorb, spawn, discard) — prevents re-clustering in subsequent sleep events
- $Q^\Omega$ initialized from the cluster's shrinkage-weighted mean **advantage** (no horizon factor, not zero) — §3.5
- K-means clustering over node embeddings; $k$ selection is an open design decision (§11)

**Ordering constraint:** sleep consolidation runs strictly after decay-based pruning in the same maintenance pass. Pruning writes graph removals; consolidation writes `parent_id` updates. Sequencing prune-first ensures consolidation never reparents a node that has simultaneously been marked for removal.

---

## 9. Retrieval

Two separate procedures at two cadences. Never merged into a single top-$k$. The hierarchy built by sleep consolidation is **active at retrieval time** — tactical retrieval is scoped to the children of the episode's active scaffold $\omega$, not a flat scan over all tactical nodes.

### 9.1 Strategic Retrieval (Once Per Episode, $d=1$ only)

$$\omega = \arg\max_{\omega_j \in \mathcal{G}_{d=1}} Q^{\Omega}_{\omega_j}(t_k)$$

Full scan over $d=1$ (small by construction). `Q_omega` holds a per-task-type **mean advantage** (§3.8), so this selects the scaffold whose episodes most beat the task-type baseline on $t_k$. No embedding step. $\omega$ serves a dual purpose: (1) conditions the agent's reasoning context for the episode, and (2) defines the retrieval boundary for all tactical selections within the episode. A scaffold with negative advantage on $t_k$ is deranked — the "penalized when returns are poor" behavior.

If $d=1$ is empty ($\omega = \text{null}$), tactical retrieval falls back to flat scan over all tactical nodes — bootstrap phase behavior only.

**Cold task type** ($Q^\Omega_{\omega_j}(t_k)$ undefined for all scaffolds): fall back to the scaffold with the highest cross-task shrinkage-weighted mean advantage $\bar{Q}^\Omega_{\omega_j}$ (§3.8).

### 9.2 Tactical Retrieval (Every Step, within $\omega$'s cluster)

At every step, tactical candidates are drawn exclusively from the children of $\omega$ — the tactical nodes parented under the active scaffold.

**Retrieval score** within the cluster:

$$\text{score}(s_i,\ \Delta t) = Q_i(t_k)$$

Ranked by per-task-type stored advantage $Q_i(t_k)$ for the current $t_k$. No embedding similarity step at retrieval time — the cluster membership (established at consolidation) already guarantees semantic coherence within $\omega$'s children. The stored advantage (mean MC return-to-go vs. baseline) is the sole ranking signal.

**Selection:**

$$a_t^\tau = \arg\max_{s_i \in \text{children}(\omega),\ |A^\tau| \leq N}\ Q_i(t_k)$$

**Why no similarity scoring at retrieval:** embedding similarity at every step shifts the computational burden from utility estimates (MC, computed once per episode at zero marginal per-step cost) to embedding comparisons (inference cost per step). Cluster membership established at consolidation time provides the semantic coherence guarantee; within-cluster Q-ranking then selects the most task-effective skill. This keeps retrieval fast and grounds selection in utility evidence rather than similarity heuristics.

**Constraints:**
1. Candidate set scoped to `children(omega)` — no cross-cluster retrieval
2. Top-$N$ cap respected (§7.2)
3. $\omega$ conditions retrieval boundary and reasoning context — never via Q-blending

**Bootstrap fallback** (when $\omega = \text{null}$, §5.2):

$$\text{score}(s_i,\ \Delta t) = d_i(\Delta t) \cdot \cos(e_i,\ e_q)$$

Decay-weighted cosine similarity over all tactical nodes — flat scan used only until first sleep consolidation populates $d=1$.

**Known gap:** if sleep consolidation assigned a skill to the wrong cluster (LLM misjudged absorb/spawn), that skill is unreachable under any $\omega$ that doesn't parent it. No cross-cluster fallback in Phase 1. Mitigation: LLM consolidation quality, decay pruning of misassigned low-utility nodes, and Phase 2 DAG extension allowing multi-parent nodes.

---

## 10. Episode Update Loop

```python
# ---- Persistent state (survives across episodes) ----
# G                 : SkillGraph
# current_step      : global monotonic step counter
# baseline_tac[t_k] : running mean terminal reward R      (tactical advantage baseline, §4.1)
# baseline_str[t_k] : running mean discounted return G_om (strategic advantage baseline, §3.8)
#
# ---- Helpers ----
def adv(node, t_k):
    # Stored advantage for this task type; if unseen, fall back to the node's
    # cross-task mean advantage so a fresh child is still selectable (§3.6).
    return node.Q.get(t_k, node._weighted_mean_utility(G.lambda_shrink))

for each episode:
    t_k = classify_task(episode)
    active_skills, episode_rewards, trajectory_buffer = [], [], []

    # ===== STRATEGIC SELECTION (once) — §9.1 =====
    omega = select_strategic_scaffold(G, t_k)          # None during bootstrap (d=1 empty)

    # ===== STEP LOOP — buffer only, NO learning (advantage is undefined until terminal R) =====
    for step t in 0 .. max_steps - 1:
        if omega is not None:
            candidates = sorted(G.children(omega), key=lambda s: adv(s, t_k),
                                reverse=True)[:N]       # rank children by stored advantage
        else:
            candidates = recall_tactical_flat(query=c_t, task_type=t_k, N_cap=N)  # bootstrap only
        a_t = candidates[0] if candidates else NULL_ACTION

        r_t, s_next = env.step(a_t)                    # intermediate r_t = 0; nonzero only at terminal
        episode_rewards.append(r_t)

        if a_t is not NULL_ACTION:                     # inline bookkeeping only
            a_t.n[t_k] = a_t.n.get(t_k, 0) + 1
            a_t.last_accessed_step = current_step
            active_skills.append((a_t, t))
            trajectory_buffer.append(StepRecord(node=a_t, step=t))
        current_step += 1

    # ============================ END OF EPISODE ============================
    T       = len(episode_rewards)
    R       = episode_rewards[-1] if episode_rewards else 0.0                 # terminal reward, in [-1, 1]
    G_om    = sum((gamma_omega ** t) * r for t, r in enumerate(episode_rewards))  # discounted episode return

    # Read baselines BEFORE updating them — score this episode against history excluding itself.
    b_tac = baseline_tac.get(t_k, 0.0)
    b_str = baseline_str.get(t_k, 0.0)

    # ----- TACTICAL: store advantage AND collect formations in one pass (§3.2 / §4.1) -----
    pending_formations = []
    for rec in trajectory_buffer:
        G_t  = (gamma ** (T - 1 - rec.step)) * R       # MC return-to-go (intermediate r = 0)
        A_t  = G_t - b_tac                             # advantage vs per-task-type baseline
        node = rec.node
        node.Q[t_k] = node.Q.get(t_k, 0.0) + alpha * (A_t - node.Q.get(t_k, 0.0))   # (1) store advantage
        node.recompute_decay_rate(G.lambda_base, G.epsilon, G.lambda_shrink)
        if A_t > theta_adv:                            # (2) formation gate
            pending_formations.append(rec)

    # ----- STRATEGIC: store advantage (§3.8) -----
    if omega is not None:
        A_om = G_om - b_str                            # negative when the episode underperforms baseline
        omega.Q_omega[t_k] = omega.Q_omega.get(t_k, 0.0) + alpha_omega * (
            A_om - omega.Q_omega.get(t_k, 0.0))
        omega.n_omega[t_k] = omega.n_omega.get(t_k, 0) + 1

    # ----- update baselines (incremental mean / EMA) -----
    baseline_tac.update(t_k, R)
    baseline_str.update(t_k, G_om)

    # ----- STAGE 2 — LLM JUDGMENT (batched) — §4.2 -----
    approved = llm_judge_formations(pending_formations, G)     # returns a subset
    for rec in approved:
        new_node = create_skill_node(rec)      # LLM summary + embedding; depth = TAU (tactical)
        G.insert(new_node, parent=G.root_id)   # hangs from root until next sleep reparents it (§5.2)
        # new_node.Q empty -> salience 0 -> max decay until first use

    # ----- GRAPH MAINTENANCE — decay-based pruning, tactical only (§7.1) -----
    for node in list(G.tactical_nodes()):
        retention = exp(-node.decay_rate * (current_step - node.last_accessed_step))
        if retention < theta_prune:
            G.remove(node)

    # ----- update dominant task type for nodes used this episode -----
    for node, _ in active_skills:
        node.task_type_dominant = argmax(node.n)

    # ----- SLEEP CONSOLIDATION TRIGGER (§8) -----
    unconsolidated = sum(1 for n in G.tactical_nodes() if not n.consolidated)
    if unconsolidated >= N_sleep:
        sleep_consolidation(G, theta_consolidate)     # §8.2
```

---

## 11. Open Problems

| Item | Status | Notes |
|---|---|---|
| Utility representation | **Confirmed** | Per-task-type mean **advantage** (return-to-go − baseline) for both tiers; decay salience uses $\max(\bar{Q}_{i,w},0)$; selection ranks by advantage |
| Content representation | **Confirmed** | LLM-generated concise summary; raw trace stored in `EpisodicMemoryBank` via `evidence_ids` |
| Clustering method (sleep consolidation) | **Confirmed** | K-means over node embeddings; $k$ selection open — sweep or elbow heuristic |
| Tactical retrieval technique | **Confirmed** | Within-cluster advantage-ranking under active scaffold $\omega$; bootstrap fallback uses decay-weighted cosine similarity over flat tactical layer |
| Embedding strategy | **Open** | Frozen LLM encoder vs. fine-tuned |
| Task type definition $t_k$ | **Open** | Benchmark-derived, clustered, or fixed taxonomy |
| Scaffold embedding strategy | **Open** | Cluster centroid vs. fresh embedding of synthesized content |
| LLM judgment prompt design | **Confirmed** | Single structured JSON action per cluster: `spawn` / `absorb` / `discard`; `summary` only for `spawn` |
| $Q^\Omega$ initialization scale | **Resolved (this revision)** | Both tiers store advantage (centered, difficulty-normalized); spawn-init is the cluster shrinkage-weighted mean advantage with no horizon factor. The $\frac{1}{1-\gamma^\Omega}$ inflation and the W3 empirical-horizon apparatus are retired (§3.5) |
| Avoidance skill formation | **Known gap** | Below-baseline episodes ($A_t \leq \theta_{\text{adv}}$) form no tactical nodes; explicit avoidance nodes deferred to Phase 2 |
| Intra-episode credit assignment | **Known gap** | Under sparse terminal reward, return-to-go has one sign per episode; the advantage gate (§4.1) is a coarse episode-level filter and cannot isolate the load-bearing step. Intra-trajectory localization is delegated to the LLM judger; a learned per-step credit model / PRM is deferred to Phase 2 |
| Task-dynamic normalization of Q for transfer | **Known gap** | $\bar{Q}_{i,w}$ conflates task-dynamic dissimilarity with skill specificity; normalization deferred to Phase 2 |
| Learned formation policy $\pi_{\text{form}}$ | **Deferred Phase 2** | Replaces advantage pre-filter with off-policy learned classifier |
| Transferability scoring + float-up | **Deferred Phase 2** | $\hat{T}$, depth differentiation within tactical layer |
| Affect/personalization graph | **Deferred Phase 2** | Volatile user-preference memory |
| Double Q-learning | **Deferred Phase 2** | Overestimation bias correction |
| Memory-quality reward bonus | **Deferred Phase 2** | $r_t^{\text{mem}} = Q_i(t_k) - \bar{Q}(t_k)$ |
| DAG extension | **Deferred Phase 2** | Multi-parent nodes |

---

## 12. Hyperparameter Summary

| Symbol | Role | Starting value | Status |
|---|---|---|---|
| $\theta_{\text{adv}}$ | Advantage pre-filter threshold (Stage 1); admits step to judger if $A_t = G_t - b(t_k) > \theta_{\text{adv}}$ | $0$ | sweep |
| $b(t_k)$ | Tactical advantage baseline: per-task-type running mean terminal reward $R$ | tracked, not swept | — |
| $b^\Omega(t_k)$ | Strategic advantage baseline: per-task-type running mean discounted return $G^\Omega$ | tracked, not swept | — |
| $\lambda$ | Base decay rate (flat tactical layer) | — | sweep |
| $\lambda_{\text{shrink}}$ | Bayesian shrinkage pseudocount for $\bar{Q}_{i,w}$, $\bar{Q}^\Omega_\omega$, and $Q^\Omega$ init | $10$ | sweep |
| $\epsilon$ | Salience floor in decay denominator (denominator is $\max(\bar{Q}_{i,w},0)+\epsilon$) | $0.01$ | sweep |
| $\theta_{\text{prune}}$ | Retention threshold for tactical node removal | — | sweep |
| $N$ | Hard tactical action space cap | — | sweep |
| $\alpha$ | Tactical advantage learning rate (EMA) | $0.1$ | sweep |
| $\alpha^{\Omega}$ | Strategic advantage learning rate | $0.1$ | sweep, independent of $\alpha$ |
| $\gamma$ | Tactical discount — attributes terminal reward to earlier steps in the return $\gamma^{(T-1)-t}R$ | $[0.9, 0.99]$ | sweep |
| $\gamma^{\Omega}$ | Strategic discount (separate from $\gamma$) | $[0.9, 0.99]$ | sweep, independent of $\gamma$ |
| $R$ | Evidence reservoir size per node | $50$ | sweep |
| $N_{\text{sleep}}$ | Unconsolidated tactical count triggering sleep | — | sweep |
| $\theta_{\text{consolidate}}$ | Minimum salience $\max(\bar{Q}_{i,w},0)$ (advantage margin over baseline) for consolidation eligibility | — | sweep |
| $k$ | Number of clusters in K-means sleep consolidation | — | sweep or elbow heuristic |

**Removed from Phase 1 (deferred to Phase 2):**
$\theta_1$, $\theta_2$, $\theta_{\text{CV}}$, $N_{\min}$, $\epsilon_{\text{hyst}}$, $M_{\text{wait}}$, $\lambda_{\text{slow}}$, $\lambda_{\text{fast}}$, $\theta_{\text{absorb}}$, $\rho$

---

## 13. Relationship to MemRL

| Aspect | MemRL | This Work (Phase 1) |
|---|---|---|
| Memory structure | Flat bank | Two-tier: $d=1$ strategic scaffolds + flat tactical layer |
| Storage backend | SQLite via SQLAlchemy (`MemoryService`) | Same; two tables: write-once `skill_representation`, mutable `skill_graph_state` |
| Skill formation | All experiences stored | Advantage pre-filter → LLM judgment → storage at episode end |
| Formation signal | LLM judgment only | Advantage (MC return-to-go vs. baseline; algorithmic, cheap) gates before LLM (semantic, expensive) |
| Retention | Recency / retrieval frequency | Ebbinghaus decay modulated by $\bar{Q}_{i,w}$ — shrinkage-weighted mean across task types; task-agnostic |
| Abstraction | None | Periodic sleep consolidation: K-means cluster surviving tactical memories → LLM returns structured `spawn` / `absorb` / `discard` action; `SkillRepresentation.content` stores summary; code computes $Q^\Omega$ → $d=1$ scaffold |
| Retrieval | Flat similarity scan over all memories | Two-tier: $\omega$ selected once by $\arg\max Q^\Omega(t_k)$; tactical candidates scoped to children of $\omega$, ranked by $Q_i(t_k)$ — no per-step embedding comparison |
| Action space | Flat, single-tier | Two-tier: strategic option (once per episode) + tactical action (every step, within-cluster) |
| Action space bound | Unbounded | Hard cap $\|A^\tau\| \leq N$ within cluster + soft decay pruning |
| Utility signal | MC terminal-reward EMA ($Q\leftarrow Q+\alpha(r-Q)$, Eq. 8) | Mean **advantage** per task type: tactical = MC return-to-go − baseline; strategic = discounted episode return − baseline |
| Decay salience | N/A | $\max(\bar{Q}_{i,w},0)$ — shrinkage-weighted mean advantage, floored at zero; below-baseline skills decay at max rate |
| Strategic scaffolds | None | Permanent $d=1$ nodes; never decay; $Q^\Omega$ = per-task-type mean advantage; initialized from cluster shrinkage-weighted mean advantage, not zero |
| LLM dependency | All memory decisions | Semantic judgment only (formation quality, consolidation synthesis); structural decisions are algorithmic |

---

## 14. Theoretical Derivation: Single-Discount Bias in $Q^\Omega$ (W4)

The spec (§2.6) claims that sharing a single discount $\gamma$ across both tiers introduces *systematic bias* in $Q^\Omega$ estimates when episodes are long, and that separate $\gamma$ (tactical) and $\gamma^\Omega$ (strategic) are therefore required. This section derives the claim formally so it is asserted, not merely stated.

**Setup.** Let an episode have length $T$. The strategic option-value accumulates the *full-episode discounted return*

$$G^\Omega = \sum_{t=0}^{T-1} (\gamma^\Omega)^t\, r_t$$

and is updated once per episode toward $G^\Omega$ (§3.8). The tactical utility is updated once per episode toward the MC return-to-go $G_t = \gamma^{(T-1)-t}R$ (§3.2) — no bootstrap. The two estimates track **different returns**: $Q^\Omega$ tracks the whole-episode option return; $Q_i$ tracks the terminal reward discounted back to node $i$'s retrieval step. The bias below concerns which discount the *strategic* target uses, and is independent of the tactical estimator.

**Claim.** Using a single shared discount $\gamma_{\text{shared}}$ for both tiers conflates two distinct quantities and biases $Q^\Omega$ whenever $\gamma_{\text{shared}}$ is chosen to suit the *tactical* (intra-episode return-to-go) regime.

**Derivation.** Under a single discount $\gamma_{\text{shared}}$:
- The tactical update requires $\gamma_{\text{shared}} \in [0.9, 0.99]$ so that the terminal reward propagates back to earlier retrieval steps (a tactical $\gamma \approx 0$ makes $G_t = \gamma^{(T-1)-t}R \approx 0$ for all non-terminal steps, concentrating all credit on the terminal step and starving earlier skills).
- The strategic target then becomes $G^\Omega_{\text{shared}} = \sum_{t=0}^{T-1} (\gamma_{\text{shared}})^t r_t$.

But the *correct* strategic target under the semi-MDP options formulation is the **model-free estimate of the option's value**, which for an option of duration $T$ should discount by the **option's own discount** $\gamma^\Omega$ over the *whole-option* trajectory, not by the intra-option per-step discount. Concretely, the semi-MDP value of an option is

$$Q^\Omega(s,\omega) = \mathbb{E}\!\left[\sum_{k=0}^{K-1} (\gamma^\Omega)^k\, R^{(k)} \;\middle|\; s_0=s,\ \omega_0=\omega\right]$$

where $R^{(k)}$ is the *cumulative reward over the $k$-th option execution* and $K$ is the number of options. In Phase 1 each episode runs a single option to termination, so $K=1$ and the strategic target is the **undiscounted** (or $\gamma^\Omega$-discounted) episode return — **not** $\sum_t (\gamma_{\text{shared}})^t r_t$.

The bias is the ratio of the two geometric sums:

$$\text{bias}(T) \;=\; \frac{G^\Omega_{\text{shared}}}{G^\Omega} \;=\; \frac{\sum_{t=0}^{T-1} (\gamma_{\text{shared}})^t r_t}{\sum_{t=0}^{T-1} r_t} \;=\; \frac{1 - (\gamma_{\text{shared}})^T}{(1-\gamma_{\text{shared}})\,T} \quad (\text{for constant } r_t)$$

For $\gamma_{\text{shared}} = 0.95$, $T = 30$ (default `max_steps`): $\text{bias} \approx \frac{1 - 0.95^{30}}{0.05 \cdot 30} = \frac{1 - 0.215}{1.5} \approx 0.52$. The single-discount estimate is **≈48% below** the undiscounted episode return that $Q^\Omega$ is supposed to track. For $T = 50$: bias $\approx 0.37$ (a 63% underestimation). The bias is **monotone decreasing in $T$** — exactly the "systematic bias for long episodes" claimed in §2.6.

**Why separate discounts fix it.** With $\gamma^\Omega$ chosen *independently* of $\gamma$, the strategic target $G^\Omega = \sum_t (\gamma^\Omega)^t r_t$ can be set to track the whole-option return directly (e.g., $\gamma^\Omega \to 1$ recovers the undiscounted return; $\gamma^\Omega = 0.99$ gives mild across-episode decay when $K>1$ in Phase 2), while $\gamma$ remains free to tune per-step tactical credit. The two hyperparameters index two distinct timescales that a single scalar cannot span.

**Empirical ablation control (implemented).** The reference implementation exposes `strategic_discount_mode`: `"separate"` (default, $\gamma$ vs $\gamma^\Omega$) vs `"shared"` (collapses $\gamma^\Omega$ onto $\gamma$, reproducing the single-discount regime above). This makes the §2.6 claim *falsifiable*: an ablation comparing the two modes on long-episode benchmarks (ALFWorld, LLB-os) should show the shared mode systematically under-estimates $Q^\Omega$ and degrades scaffold selection. If the ablation shows no difference, the claim must be retracted per Reviewer W4.

---

## 15. Relationship to Hierarchical RL Literature (W6)

The architecture reuses well-known hierarchical-RL (HRL) and skill-discovery primitives. This section maps each component to its closest HRL analogue and states what is genuinely new beyond domain transfer, so the contribution is not over-claimed.

| Component | Closest HRL analogue | What is new here (beyond domain transfer) |
|---|---|---|
| Two-tier options ($d=1$ strategic / flat tactical) | Sutton, Precup & Singh (1999) *Options*; Vezhnevets et al. (2017) *FeUdal Networks* (manager/worker) | Memory side-channel $\mathcal{M}$ conditioning the policy (not part of $S$); options are *retrieved skill scaffolds* with LLM-synthesized content, not learned sub-policies |
| Strategic option-value $Q^\Omega$ | Semi-MDP option-value (Sutton et al. 1999; Bacon et al. 2017 *Option-Critic*) | Per-task-type **advantage** storage + shrinkage-weighted salience; advantage cluster-mean initialization (not zero, avoiding FeUdal dead-layer; no horizon inflation) |
| Tactical MC utility estimation | Monte Carlo return estimation over a discrete action set (Sutton & Barto 2018, Ch. 5) | The "action set" is a *self-organizing skill graph* with utility-modulated decay controlling its membership and a hard cap $\|A^\tau\|\le N$; return-to-go from each skill's retrieval step |
| Skill discovery via clustering | Eysenbach et al. (2019) *DIAYN*; Tessler et al. (2017) *H-DRLN* (skill discovery + reuse) | Discovery is *offline batch* (sleep consolidation) over semantically meaningful LLM-summarized skills, not over latent policy states; LLM returns a structured spawn/absorb/discard decision |
| Utility-based retention | Prioritized experience replay (Schaul et al. 2016) recency/frequency heuristics | Biologically-grounded **Ebbinghaus decay modulated by $\bar{Q}_{i,w}$** — retention is a continuous function of *utility evidence*, decoupled from recency; task-agnostic global salience for a unified graph |
| Advantage formation gate | Advantage estimation (Schulman et al. 2016 GAE); prioritized replay signals | **Two-stage gate**: cheap algorithmic advantage pre-filter (MC return-to-go vs. per-task-type baseline) *before* an expensive LLM semantic judgment — explicitly offloads the *structural* "what to store" decision off the LLM |

**The genuine contribution**, beyond applying HRL primitives to LLM agents, is the **division of labor between an algorithmic structural layer and an LLM semantic-judgment layer**: MC advantage, Ebbinghaus decay, and clustering decide *formation, retention, and consolidation timing*; the LLM is invoked only for *semantic* judgment (is this a coherent skill? does this cluster generalize?). This is the opposite of base MemRL, which delegates *all* memory-quality judgment to the backbone LLM's in-context reasoning at retrieval time. The side-channel $\mathcal{M}$ formulation (memory conditions the policy without entering $S$, preserving convergence) and the advantage-gate-precedes-LLM-call pattern are the domain-specific novelty — not the options or clustering themselves, which are acknowledged HRL borrowings.

**Positioning vs Option-Critic / DIAYN:** those works *learn* the option policy and termination end-to-end from reward. This work does **not** learn sub-policies — the backbone LLM is the (fixed) policy; the options are *memory structures* that condition the LLM's context. The contribution is a memory architecture, not a new HRL algorithm, and should be framed as such.