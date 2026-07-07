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
    content     TEXT NOT NULL,      -- LLM-formatted memory text. NOT the raw EpisodicMemoryBank record.
                                    -- Kept concise for context-window efficiency at retrieval.
                                    -- Tactical: LLM-formatted episodic step trace (goal + literal
                                    -- ordered observation->action steps + outcome) -- not compressed
                                    -- into an abstracted procedure/rule.
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

**Content generation:** at tactical node creation, the LLM is called once to reformat the admitted step's experience into an episodic memory: a goal line, the literal ordered sequence of (observation, action) steps taken, and an outcome line — not compressed into an abstracted procedure or reusable rule, and not the raw `EpisodicMemoryBank` record either (the LLM still cleans up/formats the trace for storage). The raw trace itself is stored in `EpisodicMemoryBank` via `evidence_ids` and is available for inspection but never surfaced directly at retrieval. This keeps retrieved content short enough to fit within the agent's context window when multiple nodes are retrieved per step.

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

**`content` and `embedding`** live in `skill_representation`, not on `SkillNode`. For tactical nodes: LLM-formatted episodic step trace (goal + literal ordered observation→action steps + outcome, not an abstracted procedure/rule) + embedding of that trace. Raw experience trace is stored separately in `EpisodicMemoryBank` via `evidence_ids` — never surfaced directly at retrieval. For strategic nodes: LLM-synthesized abstraction from cluster summaries + embedding of that abstraction. Cluster centroid embedding is used as the initial embedding; can be replaced with a fresh embedding of the synthesized content — pick one, document it, do not leave ambiguous.

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

**Phase 2 change:** replace the pure option-value argmax with embedding similarity between the query and strategic scaffold task descriptions to shortlist the top-$k$ candidates, then let the LLM choose among them (rather than a deterministic $Q^\Omega$ argmax with no embedding step). Not yet implemented in Phase 1.

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
| Content representation | **Confirmed** | Tactical: LLM-formatted episodic step trace (goal + literal steps + outcome), not an abstracted procedure/rule. Strategic: LLM-synthesized abstraction from cluster summaries. Raw trace stored in `EpisodicMemoryBank` via `evidence_ids` |
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
| Strategic scaffold selection | **Deferred Phase 2** | Replace $Q^\Omega$ argmax (§9.1) with embedding similarity top-$k$ over scaffold task descriptions, then LLM chooses among the shortlist |

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
---

# Phase 2 — Working Notes (Planning, NOT Locked)

**Status:** agreed design directions from the P2 planning discussion, staged for brainstorming. Phase 1 (§1–§15) remains **locked and running** — nothing below modifies Phase 1 mechanics in place. Items tagged **[early-stage candidate]** may be pulled into the current run *only* if a diagnostic (§P2.6) justifies it. Two open diagnostics gate the object-abstraction and reflection-altitude decisions; do not lock those until resolved.

Design through-line to keep honest: every P2 item below (blend, reflection, warm-start, failed-episode rescue) is a mechanism for **extracting signal when the utility graph is starved** — i.e. weak backbone (4o-mini) on the 3 complex ALFWorld types where success ≈ 10% and the tactical layer never fills. This is the right thing to fix, but it makes the ablation in §P2.7 existential.

---

## P2.1 Tactical Retrieval — Similarity–Utility Blend

> **Status: implemented as a convex combination, not the additive form below.** `SkillSimilarityRetriever.tactical_retrieve` in `memrl/service/retrievers.py` scores cluster-scoped candidates as `lambda_retrieval * rank_norm(Q_i(t_k)) + (1 - lambda_retrieval) * rank_norm(cos(e_i, e_q))` (new `MemoryConfig.lambda_retrieval`, default 0.5, reasoned from MemRL's own blend but not independently swept). Rank-normalization matches the spec's normalization requirement; the combination shape was changed from `norm(Q) + λ·norm(sim)` to the convex form per explicit instruction, ahead of the §P2.6.1 gate/instrumentation sequencing above.

**Problem.** Phase 1 §9.2 selects $a_t^\tau = \arg\max_{s_i \in \text{children}(\omega)} Q_i(t_k)$. This has **no dependence on $t$ or $c_t$** — for fixed $\omega$ and $t_k$ it returns the *same* skill at every step. Context-blind by construction; this is MemRL's unstable pure-utility ablation corner. State carries $h_t, c_t$ then discards them at selection.

**Change.** Restore MemRL's two-phase (recall-then-blend) retrieval, scoped to $\omega$'s children:

$$\text{score}(s_i,\ c_t) = \text{norm}\big(Q_i(t_k)\big) \;+\; \lambda_{\text{sim}} \cdot \text{norm}\big(\cos(e_i,\ e_q)\big)$$

- $e_q$ recomputed **per step** → moves with the current subgoal → supplies the temporal discrimination across steps that $Q$ (constant in $t$) structurally cannot. This is live in steady state, not merely a bootstrap patch.
- **CRITICAL normalization (do not skip).** $Q$ is an *advantage* (centered at 0, ~half negative); $\cos$ is ~$[0,1]$ and always positive. Blending raw makes the sim term dominate every comparison → silent collapse to the pure-similarity (Voyager) corner while you think you tuned $\lambda_{\text{sim}}$. **Rank-normalize both terms over the candidate set**, or squash advantage via $\sigma(A/\tau)$, *before* combining. Do **not** copy MemRL's $\lambda=0.5$ against raw advantage space.

**Precedent.** Park et al., *Generative Agents* (UIST 2023): retrieval = recency + importance + relevance; relevance (query similarity) is included *specifically because* importance-only surfaces globally-salient-but-locally-irrelevant memories — our pure-$Q$ is their importance-only failure. Voyager (Wang et al. 2023) is the pure-similarity opposite corner. The blend is the known-good middle.

**Touches:** §3.6, §9.2. New hyperparameter $\lambda_{\text{sim}}$ (§P2.8) — set after a normalization sweep, not at 0.5.

---

## P2.2 Reflection Channel — Strategic Content Revised at Sleep

**Scope decision (agreed).** Reflection lives on the **strategic node**, not in a separate store. Scoping a "when this strategy fails" lesson to the strategy that failed is natural. But reflection **revises the scaffold's own strategic `content`** (a *rewrite*), it is **not** a parallel field. Rationale: a success-abstraction `content` + a bolted-on failure note produces a self-contradictory object the conditioning step can't cleanly use ("do X / X fails because Y"). A researcher rewrites a failing plan; they don't staple a sticky-note to it.

**Framing.** Reflection = the **update rule for strategic content**. Failure trajectory = loss; reflection = natural-language gradient; rewritten scaffold summary = parameter update. This is the TextGrad abstraction (Yuksekgonul et al., *Nature* 2025) — backprop of NL critique into a text artifact. This is cleaner and more novel than "MemRL + a reflection field."

**Trigger — NOT per-failure.** On the 3 complex types 4o-mini fails ~90% of episodes; a per-failure rewrite thrashes (content overwritten from fresh failure every episode, never stabilizes — Reflexion gets away with per-attempt because it has ~3 attempts, we have thousands of episodes). Instead:

- **Revision happens inside sleep consolidation** (§8), which is already batched, already the sole $d=1$ mutation point, and already on a cadence **decoupled from $Q^\Omega$ selection**. Batching gives contrast across failures → a generalizable lesson (this is ExpeL's insight-extraction over a *set* of trajectories, Zhao et al., AAAI 2024). Decoupling from selection kills the **derank-death loop**: a low-$Q^\Omega$ scaffold still gets revised even while deranked, so its maturing lesson isn't stranded.
- **Two passes in one sleep event, different inputs — do not overload one LLM call:**
  1. *Tactical pass* (existing §8.2): cluster surviving tactical nodes → spawn/absorb/discard.
  2. *Strategic-revision pass* (new): for each existing $d=1$ scaffold, pull its recent failure trajectories → rewrite its `content` (TextGrad-style, using the §P2.5 prompt spec).
- **Cold-type escape valve** [early-stage candidate]: a **failure-count-triggered early sleep** for a scaffold accumulating failures fast, independent of the global $N_{\text{sleep}}$ counter. Global sleep cadence may be hundreds of episodes between revisions on a low-traffic complex type — too slow to escape the cold-start floor in time.

**Precedent / threat.** CLIN (Majumder et al., 2023, *Continually Learning Language Agent*): persistent causal-abstraction memory, periodically revised, retrieved to condition future attempts, on ALFWorld-family tasks. This is **very close** to "strategic node carrying an updating reflection." Our differentiator must be the **advantage-gated, $Q^\Omega$-selected, decay-curated** version — CLIN has neither $Q^\Omega$ selection nor decay. Make that delta explicit in §13 or CLIN eats the novelty.

**Touches:** §8 (new revision pass), §8.1 (optional early-sleep trigger), §13 (MemRL/related-work delta), §P2.5 (prompt).

---

## P2.3 Warm Start — Per-Task-Type Scaffold Seeding [early-stage candidate]

**Problem (the doom loop, observed live).** Weak backbone + empty memory on a hard type → ~0 success → ~no positive-advantage episodes → empty tactical layer for that type → sleep has nothing to cluster → no scaffold ever forms → type stays at floor **forever**. This is a fixed point, not slow convergence. It is the mechanistic root of "harder task types underrepresented in strategic nodes."

**Change.** Seed **one strategic scaffold per task type at init** from an LLM zero-shot plan on the task-type *description* (no evidence required). Sanctioned by existing §5.2 ("$d=1$ empty or manually seeded… no formal gating on bootstrap-seeded $d=1$ nodes"). Effect: non-null $\omega$ from episode 1; $Q^\Omega$ starts at 0 but the seed is the only option for its type so it is selected and updated. Seeds are **training wheels** — normal sleep-consolidation absorb/spawn logic (§8.2) later replaces them with evidence-grounded scaffolds. Free, no new machinery.

**Coupling.** Warm-start = positive prior (seeded scaffold); reflection (§P2.2) = accumulating negative corrections. Together they replace the 10-episode dead zone. Reflection is the *only* conditioning signal before the graph exists, which is why the "without reflection there's no point" observation on complex types is expected.

**Touches:** §5.2 (formalize seeding), §10 (init before episode 1).

---

## P2.4 Failed-Episode Rescue — Option A (Candidate-Discovery + Empty-Q Init)

**Problem.** §4.1: under sparse terminal reward $G_t$ has the **same sign for every step**, so on a failed episode *every* step gets negative advantage — including the good actions preceding the one fatal mistake. Phase 1 discards all of them (§11 "Avoidance skill formation — Known gap"). On complex types (success ≈10%) this discards ~90% of all trajectories wholesale → the tactical layer for those types stays empty. Correct to fix; this is the same cold-start root as §P2.3.

**The trap (why the naive fix corrupts values).** MC return-to-go **cannot do intra-episode credit assignment** (§11). If you rescue a good action and store it with its episode advantage $A_t = G_t - b(t_k) < 0$, the node is *simultaneously* "worth storing" and "below baseline → salience floors to 0 → max decay → pruned before ever retrieved." Rescue mechanism and deletion mechanism in the same node. So the fix **cannot** live in the advantage math.

**Option A (committed for the first P2 pass).**
- **Failed episodes → candidate discovery only.** Extend the Stage-2 LLM judger (§4.2) to failed episodes: it identifies the subset of steps that were *locally correct despite the bad outcome* and admits those.
- **Init empty, NOT at episode advantage.** Rescued nodes enter with $Q$ **empty** (§3.5 tactical-init path), i.e. unproven: salience 0, max decay. They must **earn** positive advantage through *future successful* retrievals to survive. The failed episode is used purely to *surface a candidate*; subsequent successful usage assigns the value.
- This changes *what enters Stage 2 from failed episodes* and the *init source*, and touches **nothing** in the value-update equations. Precedent: ExpeL/Reflexion use failed trajectories as learning signal without treating the failure reward as the stored value.

**Deferred within P2 (not the first pass):**
- **Option B — real PRM** (Lightman et al. 2023): per-step score independent of episode outcome → good action in a failed episode enters *positive*. This is the principled fix and is the eventual utility signal, but on ALFWorld it needs Math-Shepherd-style automatic per-step labeling via MC rollouts (Wang et al. 2024) — a substantial, noisy sub-project on long-horizon sparse tasks. Layer on **after** A proves the mechanism helps. Do not hand-wave "we'll use a PRM."
- **Option C — HER** (Andrychowicz et al. 2017): relabel failure as success toward the goal actually achieved. **Does not transfer cleanly to ALFWorld** — goals are compositional, discrete, non-substitutable; heating the wrong object hasn't achieved *a* valid goal. Needs a partial-completion→valid-subgoal relabeler, which is a paper of its own. Reflexion's authors chose verbal reflection over HER for exactly this reason. Skip unless the relabeler is built.

**Touches:** §3.5 (rescued-node init = empty), §4.1 (gate now also emits candidates from below-baseline episodes), §4.2 (judger scope extended to failed episodes), §11 (closes the avoidance-gap row). *Option B (PRM) requires the raw per-step trace §5.3 already stores — do not drop it.*

---

## P2.5 Strategic Summary Prompt Spec (Spawn Summary + Reflection Revision)

**Diagnosis first (do not skip — see §P2.6).** "Cooling a tomato and placing" as a *strategic* summary is a symptom with two opposite causes:
- **Case 2 (prompt bug):** cluster was diverse (tomato, mug, plate) but the summarizer over-anchored on one instance → prompt fix below works.
- **Case 1 (clustering bug):** cluster was tomato-*only* because tactical embeddings are computed over the object-bearing LLM-formatted trace (§5.3), so K-means clusters by *object*, not *strategy* — the DIAYN entanglement failure (Eysenbach et al. 2019): clustering in a representation that mixes *what* with *how* fragments along the wrong axis. If this is the case, the prompt fix is a **cover-up** — the scaffold is still tomato-only and only ever selected/updated on tomato tasks, and the object-agnostic phrasing just hides it.
- **Decision:** tactical storage/embedding stays **as-is** (LLM-formatted with explicit steps; no separate object-agnostic clustering descriptor — that idea is dropped). This **bets on Case 2**. The bet is only safe if §P2.6 diagnostic 2 confirms cluster diversity. If it comes back single-object (Case 1), the strategic prompt fix is cosmetic and object-fragmented consolidation is a **carried-forward known limitation**, not something the prompt spec resolves — revisit at strategic-consolidation level only, without touching tactical storage.

**The abstraction–utility tradeoff (why "just make it general" is wrong).** Too specific ("cool the tomato") → no transfer, $Q^\Omega$ splits across object-specific scaffolds. Too general ("prepare an item and place it") → **vacuous conditioning**, constrains nothing, forfeits the strategy-conditioning lever that is the project's novel contribution. FeUdal (Vezhnevets et al. 2017) avoids the vague end by emitting directional goals in a learned space; our scaffolds are *verbal* so we are maximally exposed to vagueness. Resolution: **structural generality with procedural specificity** — abstract the *object*, keep the *procedure* concrete.

**Prompt instructions (apply to both the §8.2 spawn `summary` field and the §P2.2 reflection revision):**

1. **Object abstraction, procedure retention.** Replace specific object references (tomato, mug, apple) with role descriptors (*the target object*, *the destination receptacle*, *the tool*). Do NOT abstract the procedure — the action sequence, locations, and ordering constraints stay concrete and literal.
2. **Precondition / verification structure.** Where cluster trajectories share a precondition or failure-prone step (must be holding the object; must be at the correct receptacle; appliance open before insertion), state it as an explicit checkable condition. Highest-value content for a weak backbone.
3. **Structural, not descriptive, generality.** Generalize over *what* is manipulated, never over *how*. A summary that fits any task ("complete the objective efficiently") is a failure; a summary naming one object ("cool the tomato") is a failure. Target is between: object-agnostic, procedure-specific.
4. **Length / shape — imperative, precondition-guarded (locked choice).** Short titled strategy + ordered **imperative** procedural outline (3–6 steps), each step carrying a checkable precondition, with an instruction to skip any step whose postcondition already holds. Not a prose paragraph; not a one-liner. (The strategic scaffold is injected once at $t=0$ per §9.1, before any action is taken, so the mid-episode "re-execute step 1" hazard that afflicts imperative *tactical* injection does not apply here — imperative is the right shape for a whole-episode plan. Precondition guards: STRIPS operators, Fikes & Nilsson 1971; LLM form in Guan et al. 2023, NeurIPS.)
5. **Grounding in shared evidence.** State only procedure elements present across *multiple* cluster members. A step in one trajectory is an instance detail — omit it. This is the actual defense against "tomato" over-anchoring — BUT it only works if the cluster is diverse (Case 2). If the cluster is single-object (Case 1), "state only what's shared" faithfully yields "cooling a tomato." Hence §P2.6 must run first.

**Note on altitude.** Instruction 2 (procedural precondition detail) is the right target *only if* complex-type failures are strategic. If they are grounding/execution failures, a strategic scaffold at $d=1$ is aimed at the wrong level entirely (see §P2.6); state-conditioned tactical lessons (AutoGuide, Fu et al. NeurIPS 2024) would be the correct instrument instead.

**Touches:** §8.2 (summary prompt), §P2.2 (revision prompt). Strategic-only — tactical storage/embedding unchanged.

> **Retracted (was P2.5.1 — storage≠consumption split, imperative *tactical* procedure, hybrid embedding).** Predicated on the reading that tactical `content` is a raw trajectory the agent imitates. Corrected against the implementation: tactical `content` is already **LLM-formatted with explicitly-prompted steps**, not raw storage, and is working — tactical stays as-is. The observed post-retrieval re-query loop is handled by the existing stopping mechanism; it is *not* treated as a content-format bug. The only content-representation problem is **strategic over-specificity**, addressed by the P2.5 prompt spec above. If the re-query loop ever recurs despite the stopping mechanism, the cheap lever is **consumption-side framing** (mark injected memory as a retrospective record, not an available action — Zhou et al. 2023, *Context-faithful Prompting*, EMNLP Findings), a prompt change, not a storage redesign.

---

## P2.6 Open Diagnostics — RESOLVE BEFORE LOCKING P2.2 / P2.5

## P2.6.1 Telemetry Diagnostic Gate — RUN ON CURRENT ARCH, BEFORE ANY P2 CHANGE

**Why this runs first, alone.** The observed run (≈4k steps, 4 scaffolds) produced signals that each have a *reassuring* reading and a *concerning* reading that generate the **identical curve**. Turning on P2.1–P2.4 together would confound which intervention moved which signal — the same aggregate-confound that made the reward sawtooth uninterpretable, one level up. Sequence: (a) add the instruments below, (b) re-run current arch to get clean baselines, (c) introduce P2 changes **one at a time** against those baselines. The diagnostics are a gate, not a companion to the partial P2 run.

**The core ambiguity every strategic chart currently has.** A falling per-type $Q^\Omega$, a falling `q_omega_variance`, a shrinking reward sawtooth — each is produced *both* by convergence (estimate settling under continued updates) *and* by starvation/freeze (pair stops being selected → stops updating → variance flatlines to zero and value freezes by construction). **The disambiguator is always the same: overlay the selection count.** Falling value/variance with *rising* $n_\omega$ = health; with *flat* $n_\omega$ = death. This is the deterministic-$\arg\max$ option-starvation failure (FeUdal, Vezhnevets et al. 2017; Option-Critic, Bacon et al. 2017): a deranked option never re-accrues the experience that would revive it. §9.1's exploration-free argmax is maximally exposed.

**Required instruments (all cheap, all on current arch):**

1. **$n_\omega$ overlay on every strategic value/variance chart.** For each (scaffold, task_type): plot $Q^\Omega$, `q_omega_variance`, and $n_\omega$ on shared step axis. Resolves convergence-vs-starvation for `pick_two` (variance 2e-4 → 0 by ~3.4k: convergence iff $n_\omega$ still rising there; freeze iff $n_\omega$ plateaued). This is the single highest-value instrument.

2. **Raw $G^\Omega$ logged next to advantage.** $Q^\Omega$ stores advantage vs a *rising* baseline $b^\Omega(t_k)$, so a declining advantage (e.g. `look_at_obj` 0.62→0.42) is **ambiguous**: benign if raw return $G^\Omega$ is flat (baseline caught up) vs real regression if both fall. Currently uninterpretable — a reviewer will ask. Log per-(scaffold,type) raw $G^\Omega$ and $b^\Omega(t_k)$ alongside the stored advantage.

3. **Sawtooth-period attribution.** Overlay on `mean_reward`: (i) per-task-type split of reward, (ii) vertical markers at section boundaries, (iii) vertical markers at sleep-consolidation events. Test the ≈170-episode period against section length vs $N_{\text{sleep}}$. `scaffold_count` already shows spawns are *rare and spaced* (0→1→2→3→4 over 4k, long plateaus) → structural churn is NOT the sawtooth source → curriculum/section is the leading hypothesis. Confirm: if per-type curves are individually flat and only the aggregate sawtooths, it's curriculum and says nothing about mechanism (Agarwal et al., *Statistical Precipice*, NeurIPS 2021 — stratify before drawing mechanistic conclusions).

4. **Absorb-vs-spawn event log on `scaffold_count`.** Count only rises (4 spawns observed) — cannot tell whether the *absorb* path ever fires or whether every eligible cluster spawns. Spawn-only = accretion, not consolidation (and every spawn is permanent, §6.1 — no decay on $d=1$). Log each consolidation decision (spawn/absorb/discard) with cluster size and target. If absorb never fires, that is a consolidation-logic finding independent of P2.

5. **Frozen / imbalanced-selection audit.** At run end: `61000032`≈58, `947e5e97`≈26, `1ba086f7`≈22, `d28e4344`≈9 selections — ~6× spread, newest scaffold starved. Rich-get-richer under argmax (Matthew effect in option selection). Cross-check the flat-negative pairs (Image 2: `pick_heat_then_place`/`1ba086f7` ≈ −0.05 flat) against $n_\omega$: flat value + flat $n_\omega$ + negative advantage = a hard type permanently served by a losing scaffold = the §P2.3 cold-start doom loop made empirical, and a permanent trough in every curriculum cycle that type appears in.

**What the current run already shows (not all faults).** Positive, real, report it: 4 scaffolds spawned cleanly at spaced intervals with stable plateaus; clear selection specialization (`61000032` dominant, others carving smaller shares); Image-2 scaffolds dividing the task space along strategy lines (`1ba086f7` wins `look_at_obj`, `61000032` wins heat/clean). The hierarchy **is** forming and dividing labor. The open question is not "does structure form" (yes) but "are low-traffic scaffolds learning or starving" — answerable this afternoon with instrument 1.

**Gate criterion for starting the partial P2 run.** Instruments 1–3 must be live and one clean current-arch baseline logged. Then introduce P2 changes singly: recommended order P2.1 (blend, fixes context-blindness) → measure → P2.3 (warm-start, directly targets the starvation instrument 1/5 expose) → measure → P2.4 (rescue) → measure → P2.2 (reflection, gated on §P2.6 altitude label) last. Never two at once against an un-baselined instrument.

---

## P2.7 Thesis Risk — The Existential Ablation

Reflection + warm-start + failed-episode rescue are all **bypasses around the starved utility graph**. If, on small models / hard types, most of the gain comes from these bypasses, then the two-tier graph (strategic hierarchy + decay + consolidation) is not carrying the weight the abstract claims.

**Required ablation, run early:** *full system* vs *[MemRL + reflection + warm-start, NO strategic hierarchy / NO decay / NO consolidation]*. 
- If full > bypass-only → the graph earns its place; thesis holds.
- If full ≈ bypass-only → the contribution is "a good reflection-and-warm-start recipe for weak agents" — a fine paper, but **not** the paper the abstract claims (memory *structure* helps small models).

Let this ablation's result reshape framing rather than defending the graph because it is already built. The graph-starvation-on-hard-types-with-weak-models failure is precisely the kind of critical flaw the Phase-1 lock exempts.

---

## P2.8 New / Changed Hyperparameters (Phase 2)

| Symbol | Role | Start | Status |
|---|---|---|---|
| $\lambda_{\text{sim}}$ | Similarity weight in tactical retrieval blend (§P2.1); applied to **normalized** terms | set post normalization-sweep, **not 0.5** | sweep |
| $\tau_{\text{sq}}$ | Advantage squash temperature if using $\sigma(A/\tau)$ instead of rank-norm (§P2.1) | — | sweep (only if squash chosen) |
| $N_{\text{reflect}}$ | Failures accumulated under a scaffold before its revision pass considers it (§P2.2) | — | sweep |
| $N_{\text{fail-sleep}}$ | Per-scaffold failure count triggering early sleep, independent of $N_{\text{sleep}}$ (§P2.2, optional) | — | sweep |
| $B$ | Mini-batch size — parallel game slots stepped in lockstep, thread-pooled LLM calls (§P2.9) | endpoint-capacity bound | sweep/fix per experiment |
| decay clock | Global step count advanced at barrier, batch-size-independent (§P2.9) — not physical wall-clock | global-step | fixed choice |
| $\lambda,\ \epsilon,\ \theta_{\text{prune}}$ | **RE-SWEEP** — Phase-1 values were on a per-episode retrieval-step clock; global-step time-based decay rescales them (§P2.9) | Phase-1 values void | re-sweep |

## P2.9 Parallelism — Lockstep Mini-Batch Rollout (Match MemRL), Global-Step Decay, Sync Batch-Barrier Sleep

**The real bottleneck (diagnosed, not assumed).** "Steps take too long" = **LLM inference latency**, one call per agent step, seconds each; a 50-step episode is 50 sequential LLM calls, and 3,356 tasks sequentially is enormous wall-time. This is **I/O-bound** (waiting on the API / vLLM endpoint), NOT compute-bound. The memory update itself (MC arithmetic + a SQLite write) is trivially cheap.

**Correction — Ape-X was the wrong model, retracted.** An earlier draft here specced an Ape-X actor/drainer split with a Postgres `trajectory_queue`. That is designed for the **inverted** bottleneck: many *cheap fast* actors overwhelming a *slow GPU learner*, decoupled by async queues (Horgan et al. ICLR 2018; Espeholt et al. ICML 2018). Here the actors are *slow* (LLM) and the "learner" is *free* (arithmetic + SQLite) — there is no learner to decouple from. The distributed apparatus solved a problem we do not have and *introduced* the baseline/clock/consolidation races. Dropped entirely: no Postgres, no queue, no drainer, no off-policy staleness correction, no V-trace.

**What MemRL actually does (verified in `memrl/run/alfworld_rl_runner.py`), and what we match:**
- **Lockstep mini-batch:** `current_bs` games stepped together, `for step in range(max_steps)`; finished games drop out of `active_slots` and the batch ends early when all finish.
- **Only the LLM calls are parallel** — `ThreadPoolExecutor(max_workers=len(active_slots))` wraps `agent.act()` each step (threads, because LLM calls are I/O-bound → GIL releases on the network wait). Retrieval is likewise thread-pooled per slot at batch start. This is pure **latency-hiding**: `B` concurrent in-flight requests cost ≈ one request of wall-time, up to endpoint capacity — the point of vLLM continuous batching (Kwon et al., *PagedAttention*, SOSP 2023).
- Env stepping via Gym `AsyncVectorEnv` (subprocess-parallel).
- **Memory is mutated once, single-threaded, at the batch barrier** — `update_values` / `add_memories` sit *below* the step loop, never inside a thread pool. Retrieval (a read) is parallel; every write is serial.
- **Storage: SQLite via SQLAlchemy** (`MemoryService`). No concurrent writer → SQLite's single-write-lock is never contended. **Keep SQLite; do not move to Postgres** — Postgres was only motivated by concurrent writes we no longer have, and diverging the backend costs comparability with MemRL (§13).

**Why this dissolves every hazard from the last three iterations:** memory is read at batch start and written single-threaded at the barrier, so there is **no concurrent mutation at all**. The single-writer invariant is free (no drainer needed). Baseline "read-before-update excluding self" holds because the barrier update is sequential over the batch. The sleep race disappears. "Sync sleep at batch end" is the natural shape, not a special case.

**Decay clock — global step count (agreed), advanced at the barrier.** Use a monotonic **global step counter** as the decay clock — this is the logical/simulated time that "simulates wall-clock in the environment" without the pathologies of physical `datetime.now()` (non-reproducible across hardware; inflated by stalls). Continuous-time exponential *form* ($d=e^{-\text{rate}\cdot\Delta t}$) is kept — Ebbinghaus-consistent; cognitive architectures on the same forgetting literature (ACT-R base-level, Anderson & Schooler 1991) also use simulated, not physical, time.
- **Batch-size independence (important):** advance the clock by a unit **independent of `B`** (e.g. per committed episode, or per lockstep round), NOT by summed env-steps across slots — otherwise a node's decay depends on how many parallel slots ran, coupling retention to the parallelism degree. Fix `B` per experiment regardless.
- Schema: `last_accessed_step` stays a global step index; $\Delta t$ = global steps elapsed since last retrieval.
- **⚠ Decay hyperparameters ($\lambda, \epsilon, \theta_{\text{prune}}$) RE-SWEEP** — Phase-1 values were on a per-episode retrieval-step clock; the global-step clock rescales them.

**Sleep consolidation — synchronous, at the batch barrier (chosen, and now trivially safe).** Fires once after the mini-batch finishes and the single-threaded memory update commits — graph quiescent, no concurrent writers by construction (not by locking). Removes the double-spawn race for free.
- **Accepted cost — stragglers.** Lockstep batch ends only when all slots finish or hit `max_steps`; a 50-step complex task holds its slot to the end (MemRL drops *finished* slots from the thread pool, so idle slots don't cost LLM calls, but the *next* batch still waits). Complex types are the stragglers and the cold-start types — mitigate by batch composition (avoid mixing long-complex with short) or by `max_steps`/wall-time caps, without breaking the barrier. Global-step decay does **not** tick during the wait (it advances on committed work), so stragglers don't inflict a decay burst.

**Intra-batch staleness (inherent, mild, matches MemRL — no correction needed).** All games in a batch use memory as of batch start; a formation from game 1 isn't visible to game 5 until the next batch. Bounded by `B`, no bootstrapping involved, and it is exactly MemRL's behavior — so it is *controlled for* in the comparison rather than a confound. (This is the honest, minimal version of the equivalence concern; a full sequential-vs-batch check on a small split is still worth running once, but there is no V-trace-style correction to build.)

**Scope honesty (thesis / §13).** Parallelism is a **throughput enabler, not a contribution** — and matching MemRL's exact concurrency + storage model is the point: it keeps the comparison clean and the engineering small. Do not let infra scope creep delay the AAMAS submission.

**Touches:** §5.3 (batch-barrier single-threaded flush replaces per-episode flush; **keep SQLite**), §6.2 (Δt = global step count, batch-size-independent), §7.1/§8 (pruning + sleep run at the barrier), §10 (episode loop → lockstep mini-batch loop with thread-pooled `agent.act()`), §12/§P2.8 ($B$, decay re-sweep). MemRL parity: same ThreadPool-over-LLM + lockstep-batch + SQLite model.

---

## P2.10 Deferred Beyond Phase 2 (unchanged from Phase 1 §11 / abstract)

Affect / personalization graph (dual-graph utility⊥affect separation, 50/50 similarity–utility blend, probabilistic consolidation) remains **post-P2**. P2 as scoped here is still single-graph (utility) — it adds retrieval blend, reflection, warm-start, and failed-episode rescue, but does **not** introduce the affect axis. Sequence: land P2 utility-side fixes + the §P2.7 ablation *first*; the affect graph only makes sense once the utility graph is shown to carry weight on small models.