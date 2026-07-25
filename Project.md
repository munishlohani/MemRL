# Agent Memory with Utility-Based Skill Consolidation
## Architecture Specification
**Working Paper | Summer 2026**

---

## Abstract

A memory architecture for LLM agents that organizes skills in a two-tier hierarchical graph. The **strategic tier** ($d=1$) holds reasoning scaffolds — abstract frames selected once per episode under an options/semi-MDP formalism, with option-values stored per task type. The **tactical tier** (flat, $d=2$) holds executable skills, admitted by an advantage pre-filter (MC return-to-go vs. a per-task-type baseline), stored immediately, and retained via utility-modulated Ebbinghaus decay. Periodically, a **sleep consolidation** event clusters surviving tactical memories and uses LLM judgment to abstract them into strategic scaffolds — the sole mechanism by which $d=1$ nodes are created. Both tiers store a **per-task-type mean advantage**; decay salience reads the confidence-weighted mean advantage $\bar{Q}_{i,w}$, a task-agnostic denominator consistent with the unified (non-partitioned) graph.

**Base template:** MemRL. The architecture extends it along five axes: (1) flat memory → two-tier graph whose structure comes from utility evidence and LLM abstraction, not recency; (2) an advantage-gated tactical formation pipeline; (3) a separate sleep-consolidation pipeline for strategic scaffolds; (4) options-style credit assignment for strategic actions; (5) utility-modulated decay governing graph membership.

**Central bet.** MemRL delegates all memory-quality judgment to the backbone LLM at retrieval time. Here, structural decisions — what to form, what to retain, when to consolidate — are algorithmic (advantage, decay, clustering); the LLM is trusted only for semantic work (formation-time summarization, consolidation synthesis, strategic-shortlist choice). This reduces the LLM's burden while preserving its abstraction strength — the decisive property for small backbones.

**Utility estimator.** Monte Carlo return-to-go (no bootstrap), committed at episode end. Both tiers store a per-task-type mean advantage (return-to-go minus a per-task-type baseline); selection, decay salience, consolidation eligibility, and $Q^\Omega$ initialization all read advantage. Strategic scaffolds carry advantage against a strategic baseline, so they are penalized when their episodes underperform.

**Document scope.** §1–§15 specify the **core architecture** — the locked design a coding agent implements directly. §16 (Implementation Map) ties every construct to its module. §17 (Extensions) collects forward-looking additions that build on the core without modifying it; they are design proposals, not part of the shipped contract.

---

## Implementation Notes for the Coding Agent

Read this before touching any section.

**What the system is:** a two-tier skill graph beside an LLM agent. At episode start, select one $d=1$ scaffold $\omega$ (an option, fixed for the episode) that conditions reasoning — selection blends option-value with query similarity (§9.1). At each step, select a tactical skill from $\omega$'s children — selection blends advantage with similarity, gated by a quality threshold (§9.2). Tactical skills **grow** (advantage pre-filter → storage at episode end; the LLM only summarizes admitted candidates into a generalized procedure), **shrink** (utility-modulated decay → pruning, controlled by $\theta_{\text{prune}}$), and are periodically **abstracted** (sleep consolidation → $d=1$ scaffold). The strategic layer grows only through sleep consolidation.

**Key data structures:**
- `SkillNode` (`memrl/memory/skill_node.py`) — one per skill at any depth; strategic ($d=1$) and tactical ($d=2$) share the class, populating different fields (§5.4).
- `SkillGraph` (`memrl/memory/graph.py`) — the in-memory working set; children derived by `parent_id` query.
- `MemoryService` (`memrl/service/memory_service.py`) — SQLite/SQLAlchemy persistence, retrieval, and the sleep-consolidation entry point (§5.3, §16).
- `EpisodicMemoryBank` (`memrl/memory/episodic_bank.py`) — raw experiences, linked from nodes via `evidence_ids`.

**Execution order per episode** (`EpisodeRunner.run`, `memrl/episode/agent_runner.py`):
1. Classify task type → $t_k$.
2. **Strategic selection (once):** select a $d=1$ scaffold via blended retrieval + LLM shortlist choice (§9.1); null if $d=1$ empty.
3. Per step: tactical retrieval → execute → buffer. At episode end: MC return-to-go → utility update → advantage pre-filter → node creation for every admitted step (LLM summarizes only).
4. End of episode: update $Q^\Omega$ → graph maintenance (decay + pruning) → recompute `decay_rate` → check sleep trigger.

**Critical invariants:**
- Tactical `decay_rate` $= \lambda / (\max(\bar{Q}_{i,w},0) + \epsilon)$; recompute after every utility update. Strategic nodes always have `decay_rate = 0.0`.
- `parent_id` is the single source of truth for structure; there is no `children_index`.
- `total_accessed` is a `@property`, never stored.
- All new tactical nodes enter at $d=2$. Strategic `Q_omega` and tactical `Q` are separate and never merged.
- Bootstrap: until the first sleep event, $d=1$ is empty (or manually seeded); the agent runs with $\omega=\text{null}$ and Q-learning covers the tactical layer only.

---

## 1. Introduction and Motivation

Standard agent memory conflates three questions: which experience is worth **storing**, which stored experience is worth **keeping**, and which kept experience **generalizes**. Prior work (MemGPT, A-MEM, Voyager, SkillLib) optimizes retrieval and treats formation and consolidation as secondary. MemRL updates utility with a MC terminal-reward rule ($Q \leftarrow Q + \alpha(r-Q)$, its Eq. 8) and delegates all quality judgment to the backbone LLM — workable for frontier models, but a single undifferentiated mechanism.

This work separates the three concerns and assigns each its own mechanism:
- **Formation** — advantage gate (MC return-to-go vs. per-task-type baseline; algorithmic).
- **Retention** — utility-modulated Ebbinghaus decay (algorithmic, continuous).
- **Abstraction** — periodic sleep consolidation with LLM synthesis (batch).

The central hypothesis is that the LLM's strength is semantic judgment and abstraction, not deciding retrieval frequency, retention, or consolidation timing. Offloading those to an algorithmic layer yields a more principled and debuggable system, and the effect should be largest precisely where the backbone is weak — the regime this architecture targets.

The core architecture (§1–§15) deliberately holds several richer ideas in reserve — transferability scoring, intra-tactical depth, an affect/personalization graph, a learned formation policy, DAG multi-parent nodes, and a memory-quality reward bonus. These are collected in §17 as extensions that layer onto the core rather than alter it.

---

## 2. Problem Formulation

### 2.1 MDP
$$\mathcal{MDP} = (S,\ A^{\Omega},\ A^{\tau},\ P,\ R,\ \gamma,\ \mathcal{M})$$
$\mathcal{M}$ is a **side-channel** conditioning the policy, not part of $S$ — embedding it in $S$ grows the state space with every new skill and breaks convergence. The action space partitions into $A^{\Omega}$ (strategic options, $d=1$) and $A^{\tau}$ (tactical, flat). The formalism is a semi-MDP over $A^{\Omega}$ nested around an MDP over $A^{\tau}$ (Sutton, Precup & Singh 1999, *Options*).

### 2.2 State
$$s_t = (t_k,\ c_t,\ h_t,\ \omega)$$
$t_k$ task type (fixed within episode); $c_t$ context at $t$; $h_t$ history over the last $w$ steps; $\omega$ active scaffold (selected at $t=0$, fixed).

### 2.3–2.4 Action & Transition
$\omega \in \mathcal{G}_{d=1}$ is chosen once at $t=0$ (it conditions reasoning, with no direct env transition); $a_t^\tau \in \mathcal{G}_\tau$ is chosen every step and passed to `env.step`. Both $\omega$ and $t_k$ are invariant within an episode.

### 2.5 Reward
$$r_t = r_t^{\text{env}}$$
Per-step environment feedback. A memory-quality bonus is held out of the core reward: the general form is $r_t^{\text{full}} = r_t^{\text{env}} + \beta\, r_t^{\text{mem}}$ with $\beta = 0$ (§17).

### 2.6 Discount
$$\gamma \in [0.9,0.99],\qquad \gamma^\Omega \in [0.9,0.99]$$
$\gamma$ discounts the tactical MC return-to-go $G_t = \gamma^{(T-1)-t}R$ (terminal-reward attribution to earlier steps). $\gamma^\Omega$ governs the strategic option-value over the episode return. These are **separate hyperparameters, swept independently** — a single shared $\gamma$ biases $Q^\Omega$ on long episodes (derivation in §14). Sutton et al. (1999) likewise separate intra-option and semi-MDP discounts. The `shared` mode exists only as the single-discount ablation control (§14).

### 2.7 Memory Bank
$$\mathcal{M}_t = (\mathcal{G}_t,\ \{Q_i(t_k)\},\ \{Q^{\Omega}_j(t_k)\},\ \{b(t_k)\},\ \{b^\Omega(t_k)\},\ \lambda,\ \epsilon)$$

| Component | Description |
|---|---|
| $\mathcal{G}_t$ | Unified graph: $d=1$ strategic + flat tactical |
| $\{Q_i(t_k)\}$ | Tactical mean advantage, per task type; decay reads $\bar{Q}_{i,w}$ floored at 0 |
| $\{Q^{\Omega}_j(t_k)\}$ | Strategic mean advantage (option-value), per task type; **never merged** with $Q_i$ |
| $\{b(t_k)\},\{b^\Omega(t_k)\}$ | Per-task-type advantage baselines: discounted return-to-go from the episode's first step $G_0=\gamma^{(T-1)}R$ (tactical), discounted whole-episode return $G^\Omega$ (strategic) — both tracked in the same discounted-return units as the targets they're subtracted from (§3.2, §3.8). Tracked as an **EMA** with shared rate $\alpha_{\text{baseline}}$; the first observation sets the baseline directly. An EMA is used rather than a lifetime running mean because a $1/n$ step size grows unresponsive as episodes accumulate. |
| $\lambda$ | Base decay rate (single value; flat layer) |
| $\epsilon$ | Salience floor in the decay denominator |

$\mathcal{M}$ is updated **after each episode** from the buffered trajectory; there are no per-step commits.

---

## 3. Utility Estimation

### 3.1 Semantics
Each node stores a per-task-type **advantage** — mean return-to-go relative to the per-task-type baseline:
$$Q_i(t_k) \approx \mathbb{E}[A_i(t_k)],\qquad A_i(t_k) = G_t - b(t_k)$$
A skill scores positive only if its episodes beat the task-type average. This normalizes for difficulty and makes below-average skills negative, which decay salience reads directly. The field is named `Q` for schema continuity but holds a mean advantage throughout.

### 3.2 Tactical Utility Update — MC Return-to-Go
Committed once per episode, at the end, for every retrieved tactical node. The target is the realized advantage; there is no bootstrap:
$$Q_i(t_k) \leftarrow Q_i(t_k) + \alpha[A_t - Q_i(t_k)],\qquad A_t = G_t - b(t_k),\qquad G_t = \gamma^{(T-1)-t}R$$
Intermediate rewards are zero, so $G_t$ collapses to $\gamma^{(T-1)-t}R$ (terminal $R = r_{T-1}$). There is no neighbor-max term: abstraction edges are not transition edges. Read the baseline before updating it (§4.1) so an episode is scored against history excluding itself. This is consistent with the strategic update (§3.8), which is likewise bootstrap-free.

### 3.3 Decay Salience — Confidence-Weighted Mean
The salience denominator is task-agnostic — a shrinkage-weighted mean advantage across task types, floored at zero:
$$\bar{Q}_{i,w} = \frac{\sum_k w_{ik} Q_i(t_k)}{\sum_k w_{ik}},\qquad w_{ik} = \frac{n_{ik}}{n_{ik}+\lambda_{\text{shrink}}},\qquad \text{salience} = \max(\bar{Q}_{i,w},0)$$
**The zero-floor is required.** Advantage is centered near zero, so roughly half of nodes have $\bar{Q}_{i,w}<0$; a negative value would give a negative or exploding decay rate. Flooring maps any below-baseline skill to the maximum decay rate $\lambda/\epsilon$ — exactly the intent. Above-baseline skills decay slower in proportion to advantage. Cold-start is well-defined: for a node used only on $t_{k_0}$, the weights cancel and $\bar{Q}_{i,w} = Q_i(t_{k_0})$ from the first update. Task-local $Q_i(t_k)$ is deliberately not used, because decay governs global membership in a unified graph; a task-local value would make retention path-dependent on the last task type run.

### 3.4 Derived Signals
$$G_t = \gamma^{(T-1)-t}R \quad(\text{update target, §3.2})\qquad A_t = G_t - b(t_k)\quad(\text{Stage-1 gate, §4.1})$$
Both are computed from the buffered trajectory at episode end. There is no reward model and no TD error.

### 3.5 Initialization
**Tactical:** `Q` is seeded at creation from the advantage $A_t$ that admitted the node in the first place (Stage 1, §4.1) — $Q(t_k) \leftarrow A_t$, $n(t_k) \leftarrow 1$, for the task type it was formed under. This is not a special case: §3.3's cold-start identity already states that for a node used on only one task type, $\bar{Q}_{i,w} = Q_i(t_{k_0})$, so seeding here just makes the node's initial salience equal to the evidence that justified creating it, rather than discarding that evidence and starting at salience 0 (max decay) until some later episode happens to retrieve and re-update it. Without this, a node could sit at max decay indefinitely and never clear $\theta_{\text{consolidate}}$ (§8.1) despite already having known positive evidence.

**Strategic — spawn** (new $d=1$ via consolidation): initialize per-task-type advantage from the cluster's shrinkage-weighted mean, with **no horizon factor**:
$$Q^{\Omega}_\omega(t_k) = \frac{\sum_{j\in\text{cluster}} w_j Q_j(t_k)}{\sum_{j\in\text{cluster}} w_j},\qquad w_j = \frac{n_{jk}}{n_{jk}+\lambda_{\text{shrink}}}$$
No horizon normalization is applied: both tiers store advantage on the same difficulty-normalized scale, so a $\frac{1}{1-\gamma^\Omega}$ inflation would over-value spawned scaffolds. A spawned scaffold from a strong cluster inherits positive advantage, so it is selected and updated — the FeUdal dead-layer failure (Vezhnevets et al. 2017) is avoided without inflation. Task types unseen by any cluster member are absent from $Q^\Omega$; the cold-task fallback (§9.1) handles them.

### 3.6 Tactical Action Selection
Candidates are children of $\omega$, ranked by the blended retrieval score (§9.2). A child with no `Q[t_k]` entry is scored by its cross-task mean $\bar{Q}_{i,w}$ so it stays selectable rather than locked out.

### 3.7 Failure Handling
There is no separate failure-credit mechanism. A failed episode yields low or negative $R$ → low return-to-go → negative advantage → the stored utility is pulled down. A node repeatedly below baseline drifts negative, its salience floors to zero, and it decays out. The $\gamma^{(T-1)-t}$ factor already grades credit by recency, so an explicit failure penalty is redundant. This does mean MC return-to-go credits every step uniformly up to $\gamma^{T-t}$ — no causal localization of the load-bearing step (see §11 and §17).

### 3.8 Strategic Option-Value Update
Committed once per episode at the end. The scaffold conditions the whole trajectory and is credited by the episode's strategic advantage:
$$G^\Omega = \sum_{t=0}^{T-1}(\gamma^\Omega)^t r_t,\qquad A^\Omega = G^\Omega - b^\Omega(t_k),\qquad Q^{\Omega}_\omega(t_k) \leftarrow Q^{\Omega}_\omega(t_k) + \alpha^\Omega[A^\Omega - Q^{\Omega}_\omega(t_k)]$$
$b^\Omega(t_k)$ is the per-task-type EMA baseline of $G^\Omega$ (read before the update). It uses $\gamma^\Omega$, not $\gamma$. The cross-task summary for the cold-task fallback is:
$$\bar{Q}^\Omega_\omega = \frac{\sum_k w^\Omega_{\omega k} Q^\Omega_\omega(t_k)}{\sum_k w^\Omega_{\omega k}},\qquad w^\Omega_{\omega k} = \frac{n^\Omega_{\omega k}}{n^\Omega_{\omega k}+\lambda_{\text{shrink}}}$$

---

## 4. Tactical Formation Pipeline

Raw experiences do not directly become nodes. Both stages run **at episode end** over the buffered trajectory — the admission signal (MC return-to-go) is only defined once terminal reward is known.

```
Buffered trajectory → compute G_t per step → Stage 1: advantage pre-filter A_t = G_t − b(t_k) > θ_adv
   → (above-baseline steps) Stage 2: LLM summarization only → SkillNode created → decay prunes low-value nodes
```

### 4.1 Stage 1 — Advantage Pre-Filter (end-of-episode, batched)
$$G_t = \gamma^{(T-1)-t}R,\qquad A_t = G_t - b(t_k) > \theta_{\text{adv}} \Rightarrow \text{pass to Stage 2}$$
$b(t_k)$ is the per-task-type EMA baseline (rate $\alpha_{\text{baseline}}$; first observation sets it directly). Under sparse terminal reward, $G_t$ has the **same sign for every step in an episode** — a coarse episode-success signal. Subtracting $b(t_k)$ sharpens it to "this trajectory beat the task-type average," discarding mediocre episodes before any LLM call. The gate **cannot** isolate the load-bearing step, and nothing downstream does either: every above-baseline step is stored, so a successful $T$-step episode can store up to $T$ nodes. Formation volume is therefore high; decay is the counter-pressure that removes what fails to prove out.

### 4.2 Stage 2 — LLM Summarization (no judgment)
**There is no LLM approve/reject step.** Every candidate passing Stage 1 is stored unconditionally. The LLM's sole formation-time role is summarizing an admitted candidate into content (`TacticalSummaryWriter`, `memrl/service/formation_judger.py` — not the legacy `memrl/service/builders.py` classes, which are unused dead code from the base MemRL template). Decay already removes nodes that turn out low-value, so a judge would be a redundant second quality gate on top of Stage 1 + decay. Summarization and embedding calls are batched at episode end and committed to `skill_representation` in one write.

---

## 5. Unified Skill Graph

### 5.1 Structure
$$\mathcal{G} = (V, E),\qquad \text{depth}(s_i) = \begin{cases} 0 & \text{virtual root} \\ 1 & \text{strategic scaffold — sleep only} \\ 2 & \text{tactical — flat layer} \end{cases}$$
A tree (one parent per node); no cross-edges. There is no depth differentiation within the tactical layer — all tactical nodes are peers. `parent_id` is the single source of truth; children are derived via `SELECT node_id FROM skill_graph_state WHERE parent_id = ?`. `secondary_parents` is reserved and empty (it becomes load-bearing only under the DAG extension, §17).

### 5.2 Bootstrap
Until the first sleep event, $d=1$ is empty or manually seeded; the agent runs with $\omega=\text{null}$ and Q-learning covers the tactical layer only. There is no formal gating on bootstrap-seeded $d=1$ nodes — they undergo normal absorption once consolidation begins.

### 5.3 Storage Backend
**SQLite via SQLAlchemy** (matching MemRL's `MemoryService`). Two tables:

```sql
CREATE TABLE skill_representation (          -- write-once at creation
    node_id     TEXT PRIMARY KEY,
    content     TEXT NOT NULL,   -- LLM-formatted memory text (concise for context budget).
                                 -- Tactical: distilled but object-SPECIFIC PROCEDURE (goal +
                                 --   ordered imperative steps naming the actual object/
                                 --   receptacle/appliance, + outcome) -- not abstracted, not a
                                 --   verbatim step-by-step transcript.
                                 -- Strategic: LLM-synthesized object-agnostic abstraction from
                                 --   cluster summaries (§17.4) -- the inverse altitude of tactical.
    embedding   BLOB NOT NULL    -- np.ndarray.tobytes(); frombuffer() to read.
);

CREATE TABLE skill_graph_state (             -- mutable; updated every episode
    node_id             TEXT PRIMARY KEY,
    depth               INTEGER NOT NULL,    -- 1 strategic / 2 tactical
    parent_id           TEXT,                -- NULL only for root
    task_type_dominant  TEXT,                -- argmax_k n(t_k); dynamic
    t_create            INTEGER,
    last_accessed_step  INTEGER,
    decay_rate          REAL DEFAULT 0.0,
    consolidated        INTEGER DEFAULT 0,   -- layer 2 only; True once processed by a sleep event
    Q                   TEXT,                -- JSON dict[str,float] per-task-type advantage; tactical
    n                   TEXT,                -- JSON dict[str,int] retrieval counts; tactical
    Q_omega             TEXT,                -- JSON dict[str,float] option-values; strategic
    n_omega             TEXT,                -- JSON dict[str,int] episode counts; strategic
    evidence_ids        TEXT                 -- JSON list[str], reservoir-capped at R
);
CREATE INDEX idx_parent ON skill_graph_state(parent_id);
CREATE INDEX idx_depth  ON skill_graph_state(depth);
CREATE INDEX idx_consolidated ON skill_graph_state(consolidated);
```

**Working set:** load the relevant nodes at episode start, mutate in memory, and flush to `skill_graph_state` in one batch write at episode end.

**Content is a distilled, object-specific procedure, not a verbatim transcript.** At tactical node creation the LLM distills the experience into a reusable procedure: a goal line, an ordered list of short imperative steps, and an outcome line — all naming the actual object/receptacle/appliance involved (unlike the strategic layer, which abstracts the object away, §17.4; tactical is deliberately the concrete/specific end of that altitude tradeoff). A verbatim step-by-step transcript over-fits retrieval to the single instance a memory was formed from, so the content is a condensed procedure, not a raw trace. The raw trace is still stored in `EpisodicMemoryBank` via `evidence_ids` — inspectable but never surfaced at retrieval. Embeddings are computed once at creation over the summary; only the query embedding $e_q$ is computed at inference.

**`task_type_dominant`** is dynamic: $\arg\max_k n_{ik}$, updated at episode end (`SkillNode.refresh_task_type_dominant`). For strategic nodes it is the dominant task type across the absorbed cluster.

### 5.4 Node Schema

```python
from dataclasses import dataclass, field

@dataclass
class SkillNode:
    id: str                          # UUID; joins skill_representation.node_id
    task_type_dominant: str          # argmax_k n(t_k); dynamic
    t_create: int
    depth: int                       # 1 strategic / 2 tactical
    parent_id: str | None
    secondary_parents: list[str] = field(default_factory=list)   # DAG extension, empty in core
    last_accessed_step: int = 0

    # Tactical utility (layer 2 only)
    Q: dict[str, float] = field(default_factory=dict)   # per-task-type advantage
    n: dict[str, int]   = field(default_factory=dict)   # per-task-type retrieval counts

    # Strategic option-value (layer 1 only) — SEPARATE from Q, never merge
    Q_omega: dict[str, float] = field(default_factory=dict)  # init from cluster mean (§3.5)
    n_omega: dict[str, int]   = field(default_factory=dict)  # per-task-type episode counts

    decay_rate: float = 0.0          # λ/(max(Q̄_w,0)+ε); always 0.0 for d=1
    evidence_ids: list[str] = field(default_factory=list)    # reservoir-capped at R
    consolidated: bool = False       # layer 2 only; drives sleep trigger (§8.1)

    @property
    def total_accessed(self) -> int:
        return sum(self.n.values())  # never store separately

    def recompute_decay_rate(self, lambda_base, epsilon, lambda_shrink=10) -> None:
        """Tactical only. d=1 → 0.0. Q̄_w may be negative (advantage); floored at 0."""
        if self.depth == 1:
            self.decay_rate = 0.0; return
        salience = max(self._weighted_mean_utility(lambda_shrink), 0.0)
        self.decay_rate = lambda_base / (salience + epsilon)

    def _weighted_mean_utility(self, lambda_shrink=10) -> float:
        if not self.Q: return 0.0   # new node → max decay
        ws = sum((self.n.get(k,0)/(self.n.get(k,0)+lambda_shrink))*q for k,q in self.Q.items())
        w  = sum( self.n.get(k,0)/(self.n.get(k,0)+lambda_shrink)      for k   in self.Q)
        return ws/w if w > 0 else 0.0
```

Invariants: `depth==1 ⟹ Q empty`; `depth==2 ⟹ Q_omega empty` (enforced in `__post_init__`). `content` and `embedding` live in `skill_representation`, not on the node. A spawned scaffold's initial embedding is the cluster centroid.

---

## 6. Memory Decay (Tactical Only)

$d=1$ nodes are categorically permanent — outside the decay/pruning mechanism, not merely rate 0.

$$d_i(\Delta t) = e^{-\text{decay\_rate}\cdot\Delta t},\qquad \text{decay\_rate} = \frac{\lambda}{\max(\bar{Q}_{i,w},0)+\epsilon}$$

$\Delta t$ is global steps since `last_accessed_step` (not wall-clock). $\epsilon$ starts at $0.01$.

| Condition | Rate | Consequence |
|---|---|---|
| $\bar{Q}_{i,w} \le 0$ (below baseline or new) | $\lambda/\epsilon$ (max) | pruned quickly |
| $\bar{Q}_{i,w}$ large positive | small | retained |
| $d=1$ | 0 | permanent |
| single task type | $\bar{Q}_{i,w}=Q_i(t_{k_0})$ | cold-start well-defined |

Recompute via `recompute_decay_rate` after any Q-update, on **active nodes only** ($O(\text{active})$).

---

## 7. Memory Management

**Decay-based pruning:** $d_i(\Delta t) < \theta_{\text{prune}} \Rightarrow$ remove (tactical only; task-agnostic, uses `decay_rate` directly). **Disabled by default** (`theta_prune=None`) while strategic-scaffold formation is being debugged — pruning is a no-op and the graph grows monotonically under decay. `decay_rate` is still computed and used everywhere else (salience, selection) regardless of whether removal is active; setting `theta_prune` to a float re-enables removal.

**Action-space cap:** $|A^\tau| \le N$ — the top-$N$ tactical nodes by score are eligible at retrieval (a hard bound for convergence; the strategic $d=1$ set is small by construction).

| Mechanism | Type | Controls | Hyperparam |
|---|---|---|---|
| Decay + $\theta_{\text{prune}}$ | soft | graph membership | $\theta_{\text{prune}}$ |
| $\|A^\tau\|\le N$ | hard | retrieval eligibility | $N$ |

---

## 8. Sleep Consolidation — Strategic Scaffold Formation

The **sole** mechanism creating $d=1$ nodes after bootstrap. It is periodic and batched, running after graph maintenance, and runs in **two passes**: Pass 1 decides graph structure algorithmically (no LLM); Pass 2 is the LLM's sole remaining role in this pipeline — authoring or revising a scaffold's `content`, never deciding topology.

### 8.1 Trigger
Fires when the unconsolidated tactical count $\ge N_{\text{sleep}}$ (nodes not yet processed by any sleep event). Gating on the unconsolidated count rather than total population means it fires only on genuinely new material — pruning fluctuates the total population independently. A pre-LLM **eligibility filter** admits only nodes with salience $\max(\bar{Q}_{i,w},0) > \theta_{\text{consolidate}}$ to clustering — only skills beating their baseline by a margin. This is cheap arithmetic, no LLM.

**`build_memory`** is a separate master switch, independent of the run mode (`train` / `eval_in_distribution` / `eval_out_of_distribution` — which only selects the ALFWorld data split, not whether this is "a training run"). Gating the trigger on the run mode instead of this flag silently disables consolidation on any non-`train` split, since tactical formation and Q-updates already run regardless of mode — consolidation was the only thing behaving inconsistently. `build_memory` gates the whole write side of memory, not just sleep consolidation: when `False`, tactical formation, Q-value/baseline updates, pruning, and sleep consolidation all stop, and the graph is read-only for the episode (retrieval/selection still run every step). Set `build_memory=False` to evaluate against a fixed, already-built skill graph — typically paired with `reuse_skill_db=True` (see `memory.skill_db_path`) so the run points at that graph instead of a fresh per-run database.

### 8.2 Pass 1 — Structural Decision (algorithmic, no LLM)
Per eligible cluster, absorb into whichever existing scaffold has the highest cosine similarity between the cluster centroid and the scaffold's own embedding, if that similarity clears $\theta_{\text{absorb}}$; otherwise spawn a new scaffold if the cluster is large enough ($n_{\text{min-spawn}}$), else discard:
$$\text{action(cluster)} = \begin{cases}
\text{absorb}\big(\arg\max_j \cos(\text{centroid},\ e_{\omega_j})\big) & \max_j \cos(\cdot) > \theta_{\text{absorb}} \\
\text{spawn} & \text{else, if } |\text{cluster}| \ge n_{\text{min-spawn}} \\
\text{discard} & \text{otherwise}
\end{cases}$$
A spawn writes a **placeholder** representation immediately (empty `content`, embedding = cluster centroid so a later cosine comparison has something to test against) that Pass 2 fills in within the same sleep event — the placeholder is never actually retrievable. `consolidated` is set on every cluster's tactical nodes here (spawn/absorb/discard alike), preventing re-clustering. **Ordering:** consolidation runs strictly *after* decay-pruning in the same pass, so it never reparents a node marked for removal.

$n_{\text{min-spawn}}$ also caps the clustering step's own $k$ (§16.1, `clustering.py`'s `KMeansClusteringStrategy`): $k \leftarrow \min(k, \max(1, \lfloor n_{\text{eligible}}/n_{\text{min-spawn}}\rfloor))$. Without this, `_default_k`'s $\max(2,\lfloor\sqrt{n}\rfloor)$ can fragment a small eligible pool into clusters too small to ever clear $n_{\text{min-spawn}}$ — e.g. 2 eligible nodes always become 2 singleton clusters under the raw formula, which discard forever regardless of how many sleep events fire.

```python
def pass1_structural_decisions(eligible, scaffolds, theta_absorb, n_min_spawn):
    clusters = cluster_embeddings(eligible, {n.id: graph.get_embedding(n.id) for n in eligible})
    # K-means; k = max(2, floor(sqrt(len(eligible)))), refined by Davies-Bouldin over {k-1,k,k+1},
    # then capped so no cluster is forced smaller than n_min_spawn (see above)
    decisions = []
    for cluster in clusters:
        centroid = mean_embedding([graph.get_embedding(n.id) for n in cluster])
        best = max(scaffolds, key=lambda w: cosine(centroid, graph.get_embedding(w.id)), default=None)
        best_sim = cosine(centroid, graph.get_embedding(best.id)) if best else float("-inf")
        if best is not None and best_sim > theta_absorb:
            decisions.append(("absorb", cluster, best.id))
        elif len(cluster) >= n_min_spawn:
            decisions.append(("spawn", cluster, None))
        else:
            decisions.append(("discard", cluster, None))
    return decisions
```

### 8.3 Pass 2 — Content Authoring/Revision (the LLM's sole role here)
One LLM call per **scaffold** with changed evidence this sleep event — not per cluster: a scaffold absorbing two clusters in the same event gets one combined call, not two sequential ones where the second would overwrite the first. The call is `revise_strategy(current_summary, positive_evidence, negative_evidence) -> str`, with inputs varying by trigger:

| Trigger | `current_summary` | `positive_evidence` | `negative_evidence` |
|---|---|---|---|
| Spawn | `""` | the new cluster's contents | `[]` |
| Absorb | scaffold's existing content | the newly absorbed cluster(s)' contents | scaffold's pending failure buffer, if any |
| Reflection-only (§8.4) | scaffold's existing content | `[]` | scaffold's pending failure buffer |

This is also the fix for a bug the single-call design had: absorb previously left a scaffold's `content` untouched forever (`summary: null`), so a scaffold's text never reflected the evidence it kept absorbing. The prompt (`REVISE_STRATEGY_PROMPT`) revises in place — preserving existing steps unless evidence directly contradicts them, never regenerating from scratch — and requires corrections derived from failures to be written prescriptively ("Retrieve the target object before heating it," not "Avoid heating before retrieving"). The output is re-embedded (a stale embedding would corrupt Pass 1's cosine check in a future sleep event) and upserted; the scaffold's failure buffer is popped only once this call succeeds, so a failed LLM call leaves it intact for retry next sleep event.

```python
def pass2_content_revision(touched_scaffolds, newly_spawned, positive_evidence, graph):
    reflect_only = {w.id for w in graph.nodes_at_depth(1)
                    if w.id not in touched_scaffolds and graph.failure_buffer.get(w.id)}
    for scaffold_id in touched_scaffolds | reflect_only:
        current = "" if scaffold_id in newly_spawned else graph.get_content(scaffold_id)
        new_content = llm_revise_strategy(
            current_summary=current,
            positive_evidence=positive_evidence.get(scaffold_id, []),
            negative_evidence=graph.failure_buffer.get(scaffold_id, []))
        graph.write_representation(scaffold_id, new_content, embed(new_content))
        graph.pop_failures(scaffold_id)   # flush only after this call succeeds
```

### 8.4 Reflection Channel (failure capture)
At episode end (`EpisodeRunner._queue_failed_episode_reflections`, `memrl/episode/agent_runner.py`), every failed episode with an active strategic scaffold has a condensed trace (task description + recent history + outcome) appended to that scaffold's entry in `SkillGraph.failure_buffer` — an **in-memory, uncapped, per-scaffold** dict (`graph.record_failure` / `graph.pop_failures`), unconditionally: there is no solvability gate on top of "the episode failed and a scaffold was active." Uncapped by design: a reservoir cap on the buffer itself would introduce a recency bias into which failures survive to be seen by Pass 2, exactly what this mechanism exists to avoid. The buffer is durable only until the next sleep event's Pass 2 call for that scaffold succeeds — nothing here touches SQLite, mirroring how `pending_formations` is already in-memory-only.

This makes reflection the same mechanism as ordinary content revision, not a bolted-on field: failed episodes are just `negative_evidence` alongside spawn/absorb's `positive_evidence`, both consumed by the one `revise_strategy` prompt. There is no failure-count threshold — every scaffold with any accumulated failures gets a Pass 2 call at every sleep event it is touched by, or has pending failures for. See §17.3 for the original design rationale (TextGrad framing, relationship to CLIN/ExpeL) — this section is the shipped implementation of that proposal.

---

## 9. Retrieval

Two procedures at two cadences, never merged. The consolidation hierarchy is active at retrieval time — tactical retrieval is scoped to the children of $\omega$, not a flat scan. Both tiers use the same convex blend on **rank-normalized** terms, sharing one coefficient $\lambda_{\text{retrieval}}$:
$$\text{score}(s_i) = \lambda_{\text{retrieval}}\cdot\text{rank\_norm}(Q(t_k)) + (1-\lambda_{\text{retrieval}})\cdot\text{rank\_norm}(\cos(e_i,e_q))$$
**Rank-normalization is mandatory:** $Q$ is an advantage (centered near 0, often negative) while $\cos \in [0,1]$ and positive. Blending raw values lets similarity dominate regardless of $\lambda_{\text{retrieval}}$ — a silent collapse to the pure-similarity (Voyager) corner. A raw-value blend copied from MemRL's $\lambda=0.5$ against advantage space is therefore unsound; rank-normalization is what makes the coefficient portable.

### 9.1 Strategic (once per episode, $d=1$)
Score a top-$k$ shortlist (`strategic_k`) by the blend on $Q^\Omega_{\omega_j}(t_k)$; the LLM then chooses among the shortlist. $\omega$ (1) conditions reasoning and (2) defines the tactical retrieval boundary. There is **no quality gate** on this path — an episode must always end with an active scaffold or an explicit bootstrap-null. An empty $d=1$ falls back to a flat tactical scan (bootstrap only). A cold task type ($Q^\Omega(t_k)$ undefined everywhere) falls back to the highest cross-task $\bar{Q}^\Omega_{\omega_j}$ (§3.8).

An optional UCB1-style exploration bonus guards against deterministic-argmax option starvation (§16, FeUdal/Option-Critic) — a scaffold that wins the shortlist early can otherwise be reinforced forever while siblings never accumulate enough visits to compete. Before rank-normalization, $Q^\Omega_{\omega_j}(t_k)$ is replaced by
$$q_j = Q^\Omega_{\omega_j}(t_k) + c_{\text{ucb}}\cdot\sqrt{\frac{\ln(N+1)}{n_j+1}}$$
where $n_j$ is $\omega_j$'s selection count at $t_k$ ($n^\Omega_{\omega_j}(t_k)$, falling back to its total cross-task visit count when $t_k$ is cold, mirroring $Q^\Omega$'s own fallback) and $N=\sum_j n_j$ over the shortlist candidates. $c_{\text{ucb}}=0$ (default) recovers pure-$Q^\Omega$ ranking exactly. Strategic-only — tactical retrieval (§9.2) is unaffected.

### 9.2 Tactical (every step, within $\omega$'s children)
Candidates are drawn exclusively from the children of $\omega$, scored by the blend. $e_q$ is recomputed **per step**, supplying temporal discrimination across steps that $Q$ (constant in $t$) structurally cannot — pure-$Q$ selection is context-blind and returns the same skill every step. A tactical-only **quality gate** $\theta_{\text{retrieval}}$ drops any candidate below it, so a step can legitimately retrieve nothing rather than a bad memory.
$$a_t^\tau = \arg\max_{s_i \in \text{children}(\omega),\ |A^\tau|\le N,\ \text{score}\ge\theta_{\text{retrieval}}} \text{score}(s_i)$$
The bootstrap fallback ($\omega=\text{null}$) is decay-weighted cosine $d_i(\Delta t)\cdot\cos(e_i,e_q)$ over the flat layer. A known consequence of the tree structure: a mis-clustered skill is unreachable under any $\omega$ that does not parent it — there is no cross-cluster fallback in the core (mitigated by consolidation quality, decay, and the DAG extension in §17).

---

## 10. Episode Update Loop

```python
# Persistent: G (SkillGraph); current_step; baseline_tac[t_k]/baseline_str[t_k] (EMA, §2.7)

def adv(node, t_k):                       # stored advantage, else cross-task mean (§3.6)
    return node.Q.get(t_k, node._weighted_mean_utility(G.lambda_shrink))

for each episode:
    t_k = classify_task(episode)
    retrieved_visits, episode_rewards, trajectory_buffer = [], [], []

    omega = select_strategic_scaffold(G, t_k)          # None during bootstrap (§9.1)

    # ---- STEP LOOP: buffer only; advantage undefined until terminal R ----
    # trajectory_buffer records EVERY step's raw experience unconditionally
    # -- formation (below) is gated purely on advantage, not on whether a
    # tactical node happened to be retrieved this step. retrieved_visits
    # is the separate, retrieval-gated list used only for the Q-update of
    # EXISTING nodes -- the two must not be conflated (an earlier draft of
    # this pseudocode did, and reads as a formation-deadlock that the
    # actual implementation does not have).
    for step t in range(max_steps):
        if omega is not None:
            candidates = tactical_retrieve(G.children(omega), c_t, t_k, N)   # blended, §9.2
        else:
            candidates = recall_tactical_flat(c_t, t_k, N)                   # bootstrap
        retrieved = candidates[0] if candidates else None
        a_t = agent.act(c_t, retrieved)                 # memory conditions reasoning; the agent emits a_t
        r_t, s_next = env.step(a_t)                     # intermediate r_t = 0
        episode_rewards.append(r_t)
        trajectory_buffer.append(StepRecord(c_t, a_t, t, s_next))
        if retrieved is not None:
            retrieved.n[t_k] = retrieved.n.get(t_k, 0) + 1
            retrieved.last_accessed_step = current_step
            retrieved_visits.append((retrieved, t))
        current_step += 1

    # ============ END OF EPISODE ============
    T = len(episode_rewards); R = episode_rewards[-1] if episode_rewards else 0.0
    G_0 = (gamma**(T-1))*R if T > 0 else 0.0    # discounted return-to-go from the episode's first step
    G_om = sum((gamma_omega**t)*r for t, r in enumerate(episode_rewards))
    b_tac = baseline_tac.get(t_k, 0.0); b_str = baseline_str.get(t_k, 0.0)   # read before update

    # ---- TACTICAL: update Q of RETRIEVED nodes (§3.2) ----
    for node, step in retrieved_visits:
        G_t = (gamma**(T-1-step))*R; A_t = G_t - b_tac
        node.Q[t_k] = node.Q.get(t_k,0.0) + alpha*(A_t - node.Q.get(t_k,0.0))
        node.recompute_decay_rate(G.lambda_base, G.epsilon, G.lambda_shrink)

    # ---- TACTICAL: Stage-1 admission gate over EVERY step, forming NEW
    # nodes -- independent of retrieved_visits above (§4.1) ----
    pending_formations = []
    for rec in trajectory_buffer:
        G_t = (gamma**(T-1-rec.step))*R; A_t = G_t - b_tac
        if A_t > theta_adv: pending_formations.append((rec, A_t))

    # ---- STRATEGIC: store advantage (§3.8) ----
    if omega is not None:
        A_om = G_om - b_str
        omega.Q_omega[t_k] = omega.Q_omega.get(t_k,0.0) + alpha_omega*(A_om - omega.Q_omega.get(t_k,0.0))
        omega.n_omega[t_k] = omega.n_omega.get(t_k,0) + 1

    # Both baselines track the same discounted-return units as the targets
    # they're read against (G_t / G^Omega above) -- b_tac is G_0, the whole
    # episode's return-to-go from its first step, not the raw terminal R.
    baseline_tac.update_ema(t_k, G_0, alpha_baseline)   # EMA; first obs sets directly
    baseline_str.update_ema(t_k, G_om, alpha_baseline)

    # ---- FORMATION: LLM summarizes admitted candidates (NO judgment step, §4.2) ----
    for rec, A_t in pending_formations:
        new_node = create_skill_node(rec)              # LLM summary + embedding; depth = 2
        new_node.Q[t_k] = A_t; new_node.n[t_k] = 1     # seed from admitting evidence (§3.5) -- not empty
        G.insert(new_node, parent=G.root_id)           # refresh_decay_rate() reads the seeded Q here

    # ---- MAINTENANCE: prune, then update dominant type, then sleep ----
    for node in list(G.tactical_nodes()):
        if exp(-node.decay_rate*(current_step - node.last_accessed_step)) < theta_prune:
            G.remove(node)                              # no-op unless theta_prune is explicitly set (disabled by default, §7)
    for node, _ in retrieved_visits:
        node.task_type_dominant = argmax(node.n)
    if sum(1 for n in G.tactical_nodes() if not n.consolidated) >= N_sleep:
        sleep_consolidation(G, theta_consolidate)      # §8.2
```

---

## 11. Design Decisions and Known Limitations

The architecture makes a small number of decisions that a coding agent should treat as settled, and carries a few limitations that follow from them by design.

**Settled decisions.**
- *Utility representation:* per-task-type mean advantage on both tiers; decay salience is $\max(\bar{Q}_{i,w},0)$.
- *Content representation:* tactical content is a generalized procedure (object abstracted, procedure concrete); strategic content is an LLM abstraction over cluster summaries; the raw trace lives only in `EpisodicMemoryBank`.
- *Clustering:* K-means with $k=\max(2,\lfloor\sqrt{\text{eligible}}\rfloor)$, refined by Davies-Bouldin over $\{k{-}1,k,k{+}1\}$.
- *Tactical retrieval:* within-cluster blend under $\omega$, gated by $\theta_{\text{retrieval}}$; the bootstrap fallback is decay-weighted cosine.
- *$Q^\Omega$ initialization:* the cluster's shrinkage-weighted mean advantage, with no horizon factor.

**Known limitations** (each is a direct consequence of the MC/tree design and is addressed by an extension in §17):
- *Intra-episode credit:* sparse terminal reward gives one advantage sign per episode; the load-bearing step is not isolated (a learned per-step credit / PRM is the principled fix).
- *Avoidance-skill formation:* below-baseline episodes still form no *tactical* nodes — Stage 1 (§4.1) is unchanged. Failure-derived lessons now do reach the **strategic** tier via the reflection channel (§8.4), which revises a scaffold's content from failed episodes' traces; the tactical formation gate itself remains untouched.
- *Task-dynamic $Q$ normalization:* $\bar{Q}_{i,w}$ conflates task dissimilarity with skill specificity.
- *Cross-cluster reach:* a mis-parented skill is unreachable under a scaffold that does not parent it (no DAG in the core).

**Open modeling choices** (either resolution is compatible with the spec; pick and document):
- *Embedding strategy:* frozen encoder vs. fine-tuned.
- *Task type $t_k$:* benchmark-derived, clustered, or a fixed taxonomy (`task_type_mode` selects among `explicit` / `benchmark` / `episode`).

---

## 12. Hyperparameter Summary

Defaults reflect `MemoryConfig` (`memrl/configs/config.py`). Symbols marked "ablation knob" are inert at their default and exist to toggle a controlled comparison.

| Symbol (config field) | Role | Default |
|---|---|---|
| $\theta_{\text{adv}}$ (`theta_adv`) | Stage-1 gate: store step if $A_t = G_t - b(t_k) > \theta_{\text{adv}}$ | 0.0 |
| $b(t_k),\ b^\Omega(t_k)$ | Per-task-type advantage baselines (EMA) | tracked |
| $\alpha_{\text{baseline}}$ (`alpha_baseline`) | EMA rate for both baselines; first obs sets directly | 0.1 |
| $\lambda$ (`lambda_base`) | Base decay rate (flat layer); `decay_rate` computation is degenerate while unset | None |
| $\lambda_{\text{shrink}}$ (`lambda_shrink`) | Shrinkage pseudocount for $\bar{Q}_{i,w}$, $\bar{Q}^\Omega$, $Q^\Omega$ init | 10 |
| $\epsilon$ (`epsilon_decay`) | Salience floor in $\max(\bar{Q}_{i,w},0)+\epsilon$ | 0.01 |
| $\theta_{\text{prune}}$ (`theta_prune`) | Retention threshold; pruning is **disabled by default** (`None`) while strategic-scaffold formation is being debugged | None |
| $N$ (`tactical_action_cap`) | Hard tactical action-space cap | None |
| $\alpha$ (`alpha`) | Tactical advantage learning rate | 0.1 |
| $\alpha^\Omega$ (`alpha_omega`) | Strategic advantage learning rate (independent) | 0.1 |
| $\gamma$ (`gamma`) | Tactical discount in $\gamma^{(T-1)-t}R$ | 0.95 |
| $\gamma^\Omega$ (`gamma_omega`) | Strategic discount (separate; §14) | 0.95 |
| `strategic_discount_mode` | `separate` (default) vs. `shared` (single-discount ablation, §14) | separate |
| $R$ (`r_evidence`) | Evidence reservoir per node | 50 |
| `build_memory` | Master switch for the whole write side of memory — tactical formation, Q-value/baseline updates, pruning, and sleep consolidation — independent of run mode (§8.1); when `False` the graph is read-only (retrieval/selection still run) | True |
| `reuse_skill_db` | If `True`, run scripts open `skill_db_path` directly instead of a fresh per-run database — for evaluating against a specific, already-built skill graph (typically paired with `build_memory=False`) | False |
| $N_{\text{sleep}}$ (`n_sleep`) | Unconsolidated count triggering sleep | None |
| $\theta_{\text{consolidate}}$ (`theta_consolidate`) | Min salience for consolidation eligibility | None |
| $\theta_{\text{absorb}}$ (`theta_absorb`) | Pass 1 (§8.2): absorb into the closest scaffold if $\cos(\text{centroid}, e_\omega) > \theta_{\text{absorb}}$ | 0.75 |
| $n_{\text{min-spawn}}$ (`n_min_spawn`) | Pass 1: min cluster size to spawn when no scaffold clears $\theta_{\text{absorb}}$; smaller clusters discard. Also passed to the clustering strategy as a floor on cluster size ($k \leftarrow \min(k, \max(1, \lfloor n_{\text{eligible}}/n_{\text{min-spawn}}\rfloor))$), so a small eligible pool isn't fragmented into clusters too small to ever spawn | 2 |
| $k$ | K-means cluster count | $\max(2,\lfloor\sqrt{\text{eligible}}\rfloor)$, DB-refined |
| `cluster_strategy` | Sleep-consolidation clustering backend (`get_cluster_strategy()`): `kmeans` (implemented) or `hdbscan` (stub) | kmeans |
| $\lambda_{\text{retrieval}}$ (`lambda_retrieval`) | Advantage weight in both retrieval blends (rank-normed) | 0.5 |
| $\theta_{\text{retrieval}}$ (`theta_retrieval`) | Tactical-only quality gate on blended score | 0.0 (ablation knob) |
| $c_{\text{ucb}}$ (`ucb_c`) | Strategic-only (§9.1) UCB1 exploration coefficient added to $Q^\Omega$ before rank-norm; 0.0 recovers pure-$Q^\Omega$ ranking | 0.0 (ablation knob) |

The decay parameters $\lambda,\epsilon,\theta_{\text{prune}}$ are defined against the global-step clock (§16.2). Any values tuned on a per-episode retrieval-step clock must be re-scaled to that clock before they transfer.

---

## 13. Relationship to MemRL

| Aspect | MemRL | This Work |
|---|---|---|
| Structure | Flat bank | Two-tier: $d=1$ scaffolds + flat tactical |
| Storage | SQLite/SQLAlchemy | Same; two tables (write-once repr + mutable state) |
| Formation | All stored | Advantage pre-filter is the **sole** admission; LLM only summarizes admitted candidates |
| Retention | Recency/frequency | Ebbinghaus decay modulated by $\bar{Q}_{i,w}$ (task-agnostic) |
| Abstraction | None | Sleep consolidation: cluster → algorithmic `spawn`/`absorb`/`discard` (cosine vs. $\theta_{\text{absorb}}$) → one LLM content-revision call per scaffold → $d=1$ |
| Retrieval | Flat similarity scan | $\omega$ via blend; tactical scoped to children of $\omega$, same blend |
| Utility signal | MC terminal-reward EMA | Per-task-type mean **advantage** (return minus baseline) on both tiers |
| Strategic scaffolds | None | Permanent $d=1$; $Q^\Omega$ advantage; init from cluster mean, not zero |
| LLM dependency | All decisions | Semantic only: formation summarization, consolidation synthesis, strategic-shortlist choice |

---

## 14. Single-Discount Bias in $Q^\Omega$

Sharing one discount across tiers biases $Q^\Omega$ on long episodes. $Q^\Omega$ tracks the whole-episode return $G^\Omega = \sum_t (\gamma^\Omega)^t r_t$; tactical $Q_i$ tracks terminal reward discounted back to node $i$'s step. A single $\gamma_{\text{shared}}$ must sit in $[0.9,0.99]$ for the tactical regime (otherwise $G_t\approx 0$ for non-terminal steps, starving early skills), but then the strategic target is understated. The semi-MDP option value discounts by the option's own $\gamma^\Omega$ over the whole option; with $K=1$ option per episode, the correct target is the (near-)undiscounted episode return. For constant $r_t$ the bias is:
$$\text{bias}(T) = \frac{\sum_{t=0}^{T-1}(\gamma_{\text{shared}})^t r_t}{\sum_{t=0}^{T-1} r_t} = \frac{1-(\gamma_{\text{shared}})^T}{(1-\gamma_{\text{shared}})T}$$
At $\gamma_{\text{shared}}=0.95$: $T=30 \Rightarrow \approx 0.52$ (~48% under); $T=50 \Rightarrow \approx 0.37$ (~63% under). It is monotone decreasing in $T$ — the systematic bias for long episodes. A separate $\gamma^\Omega$ fixes it ($\gamma^\Omega\to1$ recovers the undiscounted return while $\gamma$ stays free for tactical credit). This yields a **falsifiable prediction:** with `strategic_discount_mode` set to `shared`, $Q^\Omega$ should be under-estimated and scaffold selection should degrade on long-episode benchmarks relative to `separate`. If a long-episode ablation shows no such gap, the bias argument does not bind in practice.

---

## 15. Relationship to HRL Literature

| Component | Closest HRL analogue | What is new here |
|---|---|---|
| Two-tier options | Sutton/Precup/Singh (1999); FeUdal (Vezhnevets 2017) | Memory side-channel $\mathcal{M}$ (not in $S$); options are retrieved LLM-authored scaffolds, not learned sub-policies |
| $Q^\Omega$ | Semi-MDP option-value; Option-Critic (Bacon 2017) | Per-task-type advantage + shrinkage salience; cluster-mean init (no zero, no horizon inflation) |
| Tactical MC utility | MC return estimation (Sutton & Barto Ch. 5) | Action set is a self-organizing graph with decay-controlled membership + cap $\|A^\tau\|\le N$ |
| Skill discovery via clustering | DIAYN (Eysenbach 2019); H-DRLN (Tessler 2017) | Offline batch (sleep) over LLM-summarized skills; structured spawn/absorb/discard |
| Utility-based retention | Prioritized replay (Schaul 2016) | Ebbinghaus decay modulated by $\bar{Q}_{i,w}$; task-agnostic global salience |
| Advantage formation gate | GAE (Schulman 2016) | Cheap advantage pre-filter as the sole structural "what to store" decision |

**The genuine contribution is the division of labor:** an algorithmic layer (MC advantage, decay, clustering) decides formation, retention, and timing; the LLM does semantic judgment only — the opposite of MemRL. The options and clustering are acknowledged HRL borrowings; the novelty is the side-channel $\mathcal{M}$ (memory conditions the policy without entering $S$) and the advantage-gate-precedes-LLM pattern. This is a memory architecture, not a new HRL algorithm: the backbone LLM is the fixed policy, and options are memory structures that condition its context.

---

## 16. Implementation Map

The theory above maps onto the codebase as follows. A coding agent implementing or extending a section should start from the listed module.

### 16.1 Module responsibilities

| Concern | Module | Notes |
|---|---|---|
| Node model (§5.4) | `memrl/memory/skill_node.py` | `SkillNode`, decay-rate recompute, reservoir evidence, `refresh_task_type_dominant` |
| In-memory graph (§5.1) | `memrl/memory/graph.py` | `SkillGraph`: `parent_id` structure, per-task-type baselines (EMA), `insert`/`reparent`/`remove`, `unabsorbed_tactical_count`, `failure_buffer`/`record_failure`/`pop_failures` (§8.4, in-memory, uncapped, per-scaffold) |
| Utility salience (§3.3) | `memrl/utils/q_utils.py` | `get_q_salience` / `get_q_omega_salience` (shrinkage-weighted mean advantage), `compute_mc_return_to_go` (§3.2), `compute_advantage` (§3.4), `apply_q_update`, `get_expected_option_value` |
| Persistence + orchestration (§5.3) | `memrl/service/memory_service.py` | `MemoryService`: two-table SQLite I/O, working-set load/flush, node creation, pruning, sleep entry point, `retrieve_query` |
| Retrieval blend (§9) | `memrl/service/retrievers.py` | `SkillSimilarityRetriever.tactical_retrieve` / `strategic_retrieve`; `rank_normalize`; `cosine_similarity` |
| Tactical summarization (§4.2, §5.3) | `memrl/service/formation_judger.py` | `TacticalSummaryWriter`, `TacticalSummaryDraft`, `TACTICAL_SUMMARY_PROMPT`. Note: `memrl/service/builders.py` (`ProceduralizationBuilder`/`ScriptBuilder`/`TrajectoryBuilder`/`get_builder`) is unused legacy code from the base MemRL template — not the current formation path |
| Sleep consolidation (§8) | `memrl/service/sleep_consolidation/` | `SleepConsolidationService` (`service.py`): `decide_cluster_structure` (Pass 1, algorithmic, §8.2) + `revise_strategy` (Pass 2, the sole LLM call, §8.3) + `consolidate`; `clustering.py` (`KMeansClusteringStrategy`, `HDBSCANStrategy` stub, `mean_embedding` for centroids); `prompts.py` (`REVISE_STRATEGY_PROMPT`); `types.py`; `checkpoint.py` (`SleepConsolidationCheckpoint.check_and_trigger`) |
| Episode loop (§10) | `memrl/episode/agent_runner.py` | `EpisodeRunner.run`: step loop, `_resolve_agent_turn` (agentic skill dispatch, §9), end-of-episode Q updates, formation commit, strategic selection, maintenance, metrics, `_queue_failed_episode_reflections` (reflection-channel capture, §8.4) |
| Agent (policy) | `memrl/agent/` | `BaseAgent.act()` returns a discriminated `AgentDecision` = `EnvActionDecision \| SkillInvocationDecision` (`base.py`); `MempAgent` (ALFWorld), `BCBAgent` (BigCodeBench), `CustomAgent`; `EpisodeHistory` (`history.py`) holds the tool-message conversation; `prompts.py` |
| Agentic retrieval skill (§9) | `memrl/skills/memory_retrieval.py` | `MemoryRetrievalSkill.retrieve` + `MemoryRetrievalResult` — the canonical agentic-not-RAG path: the agent emits a `memory_retrieval` `SkillInvocationDecision`, the runner services it and appends `MemoryRetrievalResult.to_tool_message` as a `tool` message; contract in `memory_retrieval_skill/SKILL.md`. Retrieval is never runner-injected |
| Env abstraction | `memrl/episode/env_adapter.py` | `EpisodeEnvAdapter` ABC — `reset`/`step`/`close`/`episode_id`/`task_type`/`is_batch`/`known_task_types`, returning `EpisodeResetResult`/`EpisodeStepResult` |
| Benchmark adapters | `memrl/envs/` | `AlfWorldEpisodeEnvAdapter`, `BCBEpisodeEnvAdapter` implement `EpisodeEnvAdapter`; `alfworld_env.py`, `base.py` |
| Episodic evidence | `memrl/memory/episodic_bank.py` | `EpisodicMemoryBank`, `EpisodicRecord` — raw traces behind `evidence_ids` |
| Config (§12) | `memrl/configs/config.py` | `MempConfig` → `MemoryConfig` / `ExperimentConfig` / `RLConfig`; YAML/JSON load |
| Providers | `memrl/providers/` | `llm.py`, `embedding.py` — backbone + encoder behind thin interfaces |

### 16.2 Execution model (parallel sampling)

Sampling is a lockstep mini-batch matching MemRL: `batch_size` games are stepped together, finished slots drop out, and the batch ends when all finish. **Only LLM calls are parallelized** — a `ThreadPoolExecutor` over `agent.act()` and retrieval hides I/O-bound latency (≈ one request of wall-time up to endpoint capacity). Environments are driven together via the adapter's batch `step`.

**Memory is mutated once, single-threaded, at the batch barrier:** reads run in parallel, writes are serial. This keeps the single-writer invariant free, so SQLite's write lock is never contended and there is no concurrent-mutation hazard. Sleep consolidation fires synchronously after the barrier commit, when the graph is quiescent by construction. Each slot gets a shallow-copied agent instance, because the agent carries per-task context in mutable instance state that would otherwise race.

The **decay clock is a global step counter**, advanced once per lockstep round independent of `batch_size` — otherwise retention would couple to the parallelism degree. The continuous-time exponential decay form is kept (base-level activation is a function of simulated, not physical, time). Intra-batch staleness — game 1's formation is invisible to game 5 until the next batch — is bounded by `batch_size`, involves no bootstrapping, and matches MemRL exactly, so it is a controlled property rather than a confound.

### 16.3 Entry points

Benchmark runners live under `run/`. Adding a benchmark means implementing an adapter against `EpisodeEnvAdapter` (in `memrl/envs/`) and pointing a thin runner at `EpisodeRunner`; the memory architecture is benchmark-agnostic. Migration status:

| Runner | Wiring | Status |
|---|---|---|
| `run/run_alfworld.py` | `EpisodeRunner` + `AlfWorldEpisodeEnvAdapter` (`memrl/envs/alfworld_episode_adapter.py`) + `MempAgent` | **Migrated** |
| `run/run_alfworld_ray.py` | same as above, Ray-parallel variant | **Migrated** |
| `run/run_bcb.py` | `EpisodeRunner` + `BCBEpisodeEnvAdapter` (`memrl/envs/bcb_episode_adapter.py`) + `BCBAgent` | **Migrated** |
| `run/run_hle.py` | legacy `HLERunner` (`memrl/run/hle_runner.py`) + `strategies.py` | **Not migrated** — no HLE `EpisodeEnvAdapter` yet |
| `run/run_llb.py` | legacy `LLBRunner` (`memrl/run/llb_rl_runner.py`) + `strategies.py` | **Not migrated** — no LLB `EpisodeEnvAdapter` yet |

Because HLE/LLB still route through the legacy runners, `memrl/service/strategies.py` (`BuildStrategy`/`RetrieveStrategy`/`UpdateStrategy`/`StrategyConfiguration`, plus `ClusterStrategy` consumed by sleep consolidation) is **not** yet dead — `strategies.py`'s Build/Retrieve/Update classes and `MempConfig.get_strategy_config()` become removable only once those two runners are ported to `EpisodeRunner` + adapters. `ClusterStrategy` / `get_cluster_strategy()` stay regardless (they select the sleep-consolidation clustering backend). The old flat-RAG `AlfworldRunner` (`memrl/run/alfworld_rl_runner.py`) is superseded by `EpisodeRunner` and is retained only for reference.

---

## 17. Extensions

These build on the core (§1–§16) without modifying it. Each targets the same failure mode: when the backbone is weak on hard task types, success is rare, positive-advantage episodes are rare, the tactical layer never fills, and the utility graph is *starved*. They are design proposals with rationale, sequenced so that no two are introduced against an un-baselined instrument.

### 17.1 Warm-start scaffold seeding
The starvation failure has a fixed point: a weak backbone with empty memory on a hard type produces ~0 success → no positive-advantage episodes → an empty tactical layer → sleep has nothing to cluster → no scaffold ever forms → the type stays at floor indefinitely. The fix is to seed one $d=1$ scaffold per task type at initialization from an LLM zero-shot plan over the type *description* (sanctioned by §5.2). This gives a non-null $\omega$ from episode 1; the seeds are training wheels, later displaced by evidence-grounded scaffolds through normal absorb/spawn.

### 17.2 Failed-episode rescue
On complex types, most trajectories are failures and are discarded wholesale, keeping the tactical layer empty. The trap is that MC return-to-go cannot do intra-episode credit: storing a rescued good action with its negative episode advantage makes the node simultaneously "worth storing" and "max-decay → pruned," so the fix cannot live in the advantage math. The workable form routes failed episodes to *candidate discovery only*: a rescued node enters with **empty $Q$** (unproven) and must earn positive advantage through future successful retrievals. A principled successor is a real per-step credit signal (PRM, Lightman 2023, with Math-Shepherd-style labeling); hindsight relabeling (HER) does not transfer cleanly to compositional discrete goals.

### 17.3 Reflection channel — **implemented, see §8.4**
Reflection is the update rule for **strategic content** — a rewrite of a scaffold's own `content`, not a bolted-on field. The framing is textual gradient descent: a failure trajectory is the loss, an NL critique is the gradient, and the rewritten summary is the update (TextGrad, Yuksekgonul, *Nature* 2025). It is **triggered inside sleep consolidation** (batched, decoupled from $Q^\Omega$ selection), not per-failure — per-failure thrashes, and batching gives cross-failure contrast that yields a generalizable lesson (ExpeL, Zhao AAAI 2024) while decoupling from selection avoids a derank-death loop. The differentiator against CLIN (Majumder 2023), which maintains persistent revised causal-abstraction memory on the same task family, is that this version is advantage-gated, $Q^\Omega$-selected, and decay-curated rather than unconditionally accumulated.

This design shipped as core §8.3–§8.4: `revise_strategy` is the single Pass-2 LLM call (spawn/absorb/reflection all route through it), and `SkillGraph.failure_buffer` is the in-memory, uncapped, per-scaffold accumulator that `EpisodeRunner._queue_failed_episode_reflections` populates and sleep consolidation flushes only on a successful revision.

### 17.4 Strategic summary altitude
The abstraction–utility tradeoff governs scaffold content: too specific ("cool the tomato") gives no transfer, too general ("prepare an item") gives vacuous conditioning. The resolution is **structural generality with procedural specificity** — abstract the object into role descriptors, keep the procedure concrete, and state shared preconditions as checkable guards (STRIPS-style; Guan 2023), in 3–6 imperative steps. Whether a $d=1$ scaffold is even the right instrument depends on failure *altitude*: if hard-type failures are grounding/execution rather than strategic, state-conditioned tactical lessons (AutoGuide, Fu NeurIPS 2024) are the better lever, which gates §17.3.

### 17.5 Diagnostics
Distinguishing convergence from starvation is the core measurement problem: a falling per-type $Q^\Omega$, falling variance, and a shrinking reward sawtooth each admit both readings and produce the same curve. The disambiguator is the **selection count $n_\omega$**: falling value/variance with *rising* $n_\omega$ is health; with *flat* $n_\omega$ it is the deterministic-argmax option-starvation death (FeUdal, Option-Critic). Recommended instruments: (1) $n_\omega$ overlaid on every strategic chart; (2) raw $G^\Omega$ and $b^\Omega$ next to advantage, so a declining advantage against a rising baseline is interpretable; (3) sawtooth attribution — per-type reward split with section and sleep markers; (4) an absorb-vs-spawn log (spawn-only is accretion, not consolidation); (5) a frozen/imbalanced-selection audit for the Matthew effect where the newest scaffold is starved.

### 17.6 The existential ablation
Warm-start, rescue, and reflection are all bypasses around the starved graph, so the graph must earn its place directly: *full system* vs. *[MemRL + reflection + warm-start, with no hierarchy, no decay, no consolidation]*. If full > bypass-only, the memory *structure* carries weight. If full ≈ bypass-only, the contribution is a good reflection-and-warm-start recipe rather than a claim about memory structure for small models — and the framing should follow the result.

### 17.7 Reserved directions
An affect/personalization axis (a dual graph with utility ⊥ affect, a 50/50 blend, and probabilistic consolidation) stays out of the single-graph design and only becomes worthwhile once the utility graph is shown to carry weight on small models. Also reserved: transferability scoring $\hat{T}$ and float-up for intra-tactical depth; a DAG with multi-parent nodes (activating `secondary_parents`) to close the cross-cluster reachability gap; a learned formation policy $\pi_{\text{form}}$; a memory-quality reward bonus ($\beta>0$ in §2.5); and double Q-learning.
