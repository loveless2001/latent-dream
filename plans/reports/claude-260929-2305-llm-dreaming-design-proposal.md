# Verified Sleep — a wake/sleep loop where dreams are *tasks*, not *memories*

Author: claude (#lab, 2026-09-29). Status: PROPOSAL, nothing run. For comparison with codex's proposal.

## TL;DR

- "Dream" = an offline phase with no new external data. The model generates new material from its own experience, and that material changes how it behaves the next day. It does 5 jobs, each borrowed from sleep neuroscience: **replay** (the day's experience plus older data mixed in), **abstract** (pull out general rules), **recombine** (join memories from different areas), **regularize** (add noise so it doesn't overfit) and **downscale/forget**. A 6th job is optional: **anticipate** (pre-solve likely next-day questions).
- As of 2026, production "dreaming" (Anthropic Managed Agents, May 2026; OpenAI Dreaming V3, Jun 2026; Claude Code auto-dream) **only curates text memory**: merge, dedupe, prune, temporal updates. It doesn't recombine, doesn't imagine and doesn't change weights. Research "sleep" (Behrouz+ 2606.03979, Lee+ 2605.26099, Zahn+ 2607.16256) does change weights. By its own admission it has **no safeguard against self-reinforcing errors**, no scheduling policy, and effects that are fragile.
- My thesis: **dream in the model's head, check in the world.** Dreams are generated as *tasks*. Only a dream whose outcome passes a world-grounded verifier (interpreter, env, test, checker) can touch weights. A dream that can't be checked becomes a *hypothesis* to test during the next wake, and is never stored as a fact.
- **Typed routing by memory type:** facts go to an editable text store, never weights (Gekhman 2024: FT on new facts ↑ hallucination). Skills and principles go to weights. Unverified content goes to quarantine. Every item carries a **provenance tag** (reality monitor).
- **Sleep can't make you worse:** a morning gate checks previously-correct real cases, anchors and the false-memory rate, and rolls back the night's weights if any check fails. The in-house memsub lesson applies directly: aggregate ↑ while protected cases are still lost.
- Falsifiable plan: a 9-arm ladder on a contamination-free, test-verified "evolving private library" world. The key question (H3) is whether weight-consolidation beats text-only curation at matched wake context. If it doesn't, dreaming should stay non-parametric at this scale. Kill criteria below.

## 1. What the evidence says (research digest)

| Source | Finding | Design consequence |
|---|---|---|
| Lewis, Knoblich, Poe 2018 (TICS) | NREM replay abstracts gist/rules within a schema; REM links across schemas; interleaved NREM↔REM cycles build and restructure frameworks | Night = K cycles, NREM-heavy early and REM-heavy late |
| Hoel 2021 (Patterns), Overfitted Brain | Dreams = sparse, corrupted, hallucinatory inputs that prevent overfitting to correlated daily experience | Noise dreams as a *control arm* (is the gain regularization or recombination?) |
| Mattar & Daw 2018 | Replay is prioritized by gain × need | Triage score for what to replay |
| Tononi/Cirelli SHY | Sleep downscales synapses; only repeatedly reinforced traces survive | Nightly adapter norm budget + decay |
| McClelland+ 1995 CLS | Fast hippocampus + slow cortex; interleave old with new | Episodic store + weights; anchor-interleaved replay |
| Zahn, Evans, Eagleman 2607.16256 | Within-domain rehearsal **null** (−1.8±4.4pp). Cross-domain recombination +5.64±2.31pp (Llama-8B, 5/5 seeds) **only at LoRA r≥256**; null at r=128 and at 70B/72B. Same material in-context doesn't reproduce it | Recombination is plausible but fragile → must be tested head-on; preregister a rank sweep |
| Behrouz, Hashemi, Mirrokni 2606.03979 "LMs Need Sleep" | Fast→slow weight consolidation (distillation + RL imitation) + dreaming via stochastic MoE routing. Gaps they admit: no safeguard vs self-reinforcing errors/alignment drift, unbounded expert growth, no sleep trigger | Gate + provenance + norm budget + sleep-pressure scheduler fill exactly these gaps |
| Lee, McLeish, Goldstein, Fanti 2605.26099 | Offline recurrent passes write context into fast weights; longer sleep → bigger gains on deeper-reasoning items | Sleep compute scales usefully; architecture-level option for later |
| SCM 2604.20943 | Graph memory with NREM Hebbian + REM random walks + forgetting; **REM had no measurable effect** on factual recall | REM won't help recall; test it where recombination is needed (cross-domain) |
| 2606.04703 (experience internalization) | Repeated internalization cycles **collapse instead of compound**. Fixes: principle-level > instance-level; step-wise injection > global; off-policy verified teacher trajectories > on-policy context-distillation | NREM distils principles at decision points, from verified traces |
| 2607.01763 "Denser ≠ Better" | Dense on-policy self-distillation forgets more than GRPO; a drifting EMA teacher causes collapse loops. Fix: restart-and-freeze teacher, sparse rewards | Teacher = **frozen snapshot per night**; dream learning via sequence-level RL |
| SDFT 2601.19897 | A demonstration-conditioned self-teacher (on-policy) forgets less than SFT | Context-conditioned self-teacher, frozen nightly |
| Gekhman+ 2024 (EMNLP) | Unknown facts are learned slowly and **linearly ↑ hallucination** once learned | Facts are not written to weights by default |
| Lin+ 2510.15103 Sparse Memory FT | At matched new-knowledge acquisition, NQ F1 drop: full FT 89%, LoRA 71%, sparse memory slots 11% | Later option: facts go to sparse memory slots, not LoRA |
| Sleep-time compute 2504.13171 | Offline pre-reasoning on context: ~5× less test-time compute at equal accuracy; +13%/+18% (Stateful GSM-Symbolic/AIME) | Anticipatory dreaming = cheap, proven win |
| Model collapse (Shumailov 2024) / accumulate-don't-replace (Gerstgrasser 2024) | Training on your own outputs degrades the model unless real data is accumulated | Episodic real data append-only; dream share capped |
| Reality monitoring in LLMs 2607.23927 | LLM source attribution (self vs external) is fragile under episodic delay | Provenance must be explicit metadata; LLM recall of where something came from isn't reliable |
| In-house memsub (memory/CONTEXT-POLICY.md, 2026-09-11) | Reader LoRA continuation: aggregate 155/160 but still lost protected cases (7 regressions vs v2) | Gate on **per-case protected preservation**, not just the aggregate score |

## 2. Design principles

1. **Dreams are tasks, not memories.** The generator proposes problems. The world grades them. Model beliefs are never the ground truth.
2. **Typed routing.** Facts go to the text store (editable, auditable, deletable). Skills and principles go to weights. Unverified content goes to quarantine as a hypothesis.
3. **Provenance on everything.** `observed | inferred | dream-verified | dream-hypothesis`, shown to the wake model at retrieval.
4. **Frozen nightly teacher plus sparse reward.** This gives the stability fix for self-distillation for free, because every night the teacher restarts as a frozen copy.
5. **Accumulate, never replace** real episodes. Dream data is capped as a fraction of each night's training mix.
6. **Monotone by construction.** The morning gate either accepts or rolls back. The gate's exam is disjoint from anything the dream loop can see (Goodhart firewall).

## 3. Architecture

```
            WAKE                                   SLEEP (offline, no new external data)
 ┌──────────────────────────┐        ┌───────────────────────────────────────────────────────┐
 │ LLM (base + cortex LoRA) │        │ 0 TRIAGE  priority = gain × need                      │
 │  + retrieval from S      │──log──▶│ K cycles:  NREM ──▶ REM   (ratio 3:1 → 1:3 over night)│
 └──────────┬───────────────┘        │ 4 HOMEOSTASIS  norm budget, decay, prune S            │
            │                        │ 5 CONSOLIDATE  frozen teacher → cortex LoRA           │
   E: episodic store (append-only)   │ 6 MORNING GATE accept | rollback                      │
   S: semantic text store (typed,    └───────────────────────────────────────────────────────┘
      provenance-tagged)
   Q: hypothesis quarantine → becomes wake-time probes
   P: procedural = cortex LoRA (option: sparse memory slots for facts)
```

**0 Triage (hippocampal tagging).**
- `gain` = log p(y*|x, hint/correction) − log p(y*|x). This is the teacher–student gap, i.e. how much learning the episode would change behavior. For unlabeled episodes, use failure/correction/verifier-disagreement/high-entropy flags.
- `need` = recurrence estimate: cluster frequency over the last N days plus an explicit user/agent tag (targeted memory reactivation).
- Only the top-k by gain×need get replayed. Everything else stays in E.

**1 NREM (replay + abstract).** Per prioritized episode:
- (a) *transformed* replay, never verbatim: paraphrase, reversal, step-wise decision points (state → correct next action), counterfactual "what would have avoided the error";
- (b) gist across episode clusters: "what general rule covers these k cases?" gives principle candidates;
- (c) housekeeping on S (the production-dreaming ops): merge, dedupe, resolve contradictions (newer observed > older), update temporal facts;
- (d) facts go to S (`observed`, source ids); principles go to the consolidation queue.

**2 REM (recombine + rehearse in the world).**
- Seeds: e1 ~ priority. e2 comes from a **different** schema cluster, sampled from the mid-to-far embedding-distance quantile.
- Operators:
  - *bridge*: a task needing principle(e1) ∘ principle(e2);
  - *mutate*: change constraint, scale or edge case;
  - *threat-sim*: target the model's logged failure modes;
  - *noise*: Hoel corruption, used only as a control or a small fraction.
- **World check:** the dream carries inputs; the ground truth comes from the environment (execute the real library/interpreter/simulator), not the model. Execution-based agreement of independently sampled solutions → reward.
- Keep only *learnable* dreams: pass-rate in [0.2, 0.8] over n samples. Trivial and impossible dreams are dropped, which blocks the "invent easy-to-verify dreams" hack.
- A dream the world can't check becomes `dream-hypothesis` in Q, is surfaced as a probe in the next wake ("test this if the chance arises"), and is never consolidated.
- Anticipatory pass (sleep-time compute): predict likely next-day queries from recent context and pre-derive answers, which go to S as `inferred`.

**4 Homeostasis (SHY).**
- The cortex LoRA has a norm budget B. Each night it is downscaled by γ; only traces reinforced again by replay regain magnitude, so unused skills fade gracefully (desired under drift).
- S is pruned by importance × e^(−λ·age) × (1+access). Pruned items are **archived**, never deleted from E.

**5 Consolidation.**
- Teacher = frozen snapshot of today's model **with** the principle/episode in context. Student = the same model **without** it.
- Principles → off-policy distillation on *verified-correct* teacher trajectories at decision points (2606.04703 recipe).
- Verified dreams → GRPO with the world reward (sparse, low-drift).
- The mix always includes anchor replay from past real episodes plus a general-capability anchor set, plus KL-to-base on anchor prompts. The dream share is capped (e.g. ≤50%, TBD by pilot).

**6 Morning gate** (only real, held-out, never-dreamed items). Accept the night's weights iff all four hold:
- (i) target slices non-negative with lower CI > −δ;
- (ii) protected set (all previously-correct real items, greedy decode) losses ≤ the measured noise floor of re-running yesterday's model;
- (iii) false-memory probes (non-existent APIs/facts, dream-hypotheses asked as facts) not worse than yesterday;
- (iv) general/safety anchors within the noise floor.

If any check fails, roll back the weights and keep only the S/Q updates that passed their own checks.

**Scheduler (Borbély two-process).**
- Sleep pressure S_t = Σ priority of unconsolidated episodes + S-store bloat + contradiction count. Circadian C = idle/cheap-compute window.
- **Nap** = text-only housekeeping (cheap) when bloat is high. **Full night** = NREM/REM + weights when S_t > θ.

## 4. Testbed — "evolving private library" world

- **Why:** it's contamination-free, has a cheap exact verifier (hidden unit tests + execution), and both facts (API signatures/conventions) and skills (usage patterns) are learnable in it.
- **World:**
  - A synthetic Python library with 3 domains (e.g. `geo`, `ledger`, `text`). Each has non-standard conventions absent from pretraining (1-indexed ranges, custom date encoding, sentinel errors).
  - On days 4 and 7, selected conventions change (drift tests unlearning of stale memory).
- **Day:** 40 tasks, mixed domains, agent has repo + interpreter.
- **Morning exam** (disjoint generator seeds, never visible to sleep):
  - (a) repeat-structure new instances;
  - (b) **cross-domain** tasks needing conventions from 2 domains;
  - (c) retention: day-1..t−1 items;
  - (d) drift probes;
  - (e) false-memory probes;
  - (f) anchors (HumanEval+/GSM8K/IFEval slices + refusal set).
- Model: Qwen3-8B + cortex LoRA; rank sweep {64, 256} (Zahn capacity gate). 10 nights × 3 seeds.

## 5. Preregistered ablation ladder

| Arm | Sleep content |
|---|---|
| A0 | none (day context only) |
| A1 | text-only curation (≈ production dreaming) + retrieval |
| A2 | A1 + verbatim replay SFT into LoRA (naive) |
| A3 | A1 + NREM (transformed replay, principles, frozen teacher, anchors) |
| A4 | A3 + REM within-domain mutate/threat dreams (world-verified, GRPO) |
| A5 | A3 + REM cross-domain bridge dreams (world-verified) — **full design** |
| A6 | A5 without the world verifier (self-judged dreams) |
| A7 | A3 + Hoel noise dreams only (verified) |
| A8 | A5 without the morning gate |

**Hypotheses:**
- **H1** A1 > A0 on repeat + drift (sanity; production claim).
- **H2** A3 > A2 on repeat-structure and retention (principle > instance).
- **H3 (load-bearing)** A3 > A1 on repeat-structure at a matched wake context budget. This asks whether weights add competence beyond text memory.
- **H4** A5 > A4 on cross-domain (b), ≈ elsewhere. This is the recombination claim (Zahn replication at our scale).
- **H5** If A7 ≈ A5 on (b), the gain is regularization, not recombination.
- **H6** A6: false-memory rate ↑ and a downward trend over nights vs A5 (collapse/self-reinforcement).
- **H7** A8: ≥1 seed has a regression night A5's gate would have rejected; lower cumulative retention.

**Kill / stop rules:**
- If H3 fails at both ranks, parametric dreaming is NO-GO at 8B on this world. Ship A1 plus anticipatory dreaming only, and report the negative result.
- If H4 fails and H5 shows A7 ≈ A5, drop recombination and keep regularization dreams.
- If A5's gate rejects > 50% of nights, the consolidation recipe is broken. Stop and diagnose before more nights.

**Staging (compute):**
- Phase 0: build world + exam + 1-night pilot to measure real cost (rough guess 1–3 H100-h/training-night at 8B LoRA — unverified).
- Phase 1: A0–A3 (A0/A1 need no training).
- Phase 2 **only if H3 passes**: A4, A5, A7 × 3 seeds; A6/A8 as 1-seed stress tests.

## 6. Risks

- **Self-generated-curriculum reward hacking** (dream generator = solver). Mitigations: world ground truth, learnability band, diversity quota, and a gate exam the dream loop can't see.
- **Recombination may be capacity-gated** (Zahn: r≥256 at 8B, null at 70B). Result could be a null for reasons of scale → preregistered rank sweep.
- **Facts-in-text routing** leaves wake-time context cost. If S grows large, move facts to sparse memory slots (SMF) — later phase.
- **Gate noise floor:** sampling flips can masquerade as regressions → greedy decode + measured flip rate of the unchanged model.
- **Alignment drift** across nights: refusal/safety anchors in the gate; dreams may not target policy/safety behavior.
- **Testbed validity:** a synthetic library favors procedural learning. Results may not transfer to conversational/personal memory, where no offline world exists (there, only A1 + hypothesis quarantine applies).

## 7. First concrete step (if chosen)

Phase 0 only: world generator + hidden tests + morning exam + A0/A1 harness, then a 1-night A3 pilot to measure cost and the gate noise floor. No multi-night runs until the exam is frozen and reviewed.

## Sources

- Lewis, Knoblich, Poe 2018 — https://pubmed.ncbi.nlm.nih.gov/29776467/
- Hoel 2021 Overfitted Brain — https://arxiv.org/abs/2007.09560
- Zahn, Evans, Eagleman 2026 Discovery by Dreaming — https://arxiv.org/abs/2607.16256
- Behrouz, Hashemi, Mirrokni 2026 LMs Need Sleep — https://www.emergentmind.com/papers/2606.03979
- Lee, McLeish, Goldstein, Fanti 2026 Do LMs Need Sleep? — https://arxiv.org/abs/2605.26099
- SCM 2026 — https://arxiv.org/html/2604.20943v1
- Experience internalization collapse 2026 — https://arxiv.org/abs/2606.04703
- Denser ≠ Better 2026 — https://www.alphaxiv.org/abs/2607.01763
- SDFT 2026 — https://arxiv.org/abs/2601.19897
- Gekhman+ 2024 — https://arxiv.org/abs/2405.05904
- Sparse Memory FT 2025 — https://arxiv.org/abs/2510.15103 ; follow-up https://arxiv.org/html/2605.03229v2
- Sleep-time compute 2025 — https://arxiv.org/abs/2504.13171
- Reality monitoring in LLMs 2026 — https://arxiv.org/html/2607.23927
- SEAL 2025 (self-edits; forgetting under repeated edits) — https://arxiv.org/abs/2506.10943
- Anthropic Managed Agents dreaming (May 2026) — https://claude.com/blog/new-in-claude-managed-agents
- OpenAI Dreaming (Jun 2026) — https://openai.com/index/chatgpt-memory-dreaming/
- Turing Post FOD#155 overview — https://www.turingpost.com/p/continual-learning-llms-ai-models-sleep
- Background, cited from memory, not re-fetched: Mattar & Daw 2018; Tononi & Cirelli SHY; McClelland+ 1995 CLS; Shumailov+ 2024; Gerstgrasser+ 2024; DreamCoder (Ellis+ 2021); Dreamer (Hafner).

## Unresolved questions

1. What is the target setting: agent skill learning (my testbed) or personal/conversational memory (no offline world, so the design reduces to A1 + quarantine)?
2. Is the weight budget in scope (Modal H100), or should this stay non-parametric?
3. Should the gate's protected-set rule reuse memsub's certification machinery? memsub is paused, so this only matters if reopened.
4. How should the dream share cap and the NREM:REM schedule be set — preregister fixed values or sweep in the pilot?
5. Reconcile with codex's proposal: merge into one ladder, or run them as competing arms?
