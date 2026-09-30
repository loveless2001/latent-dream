# KV dreaming — 8B, generous budget (survey-8b-generous-v1)

2026-09-30, by claude. Requested by the user (#lab 20652).

**Setup:**
- Model: Qwen3-8B-Base@49e3418, fp32, L40S. App `ap-1NHXY1UN5RgCL75sCsE86r`.
- Evict policy; cache **2048** (recent 1024); cap **8192** tokens; T=1, stream 101 (the empty start varies streams 101/202/303/404).
- 8B calibration `/runs/calib/qwen3-8b-base-cuda-v2.pt`. Independent 8B audit passed (app `ap-wEe1VKOjyqGdBAGCVuhupv`).
- 20/20 runs OK. Texts are in `…-8b-generous-survey/texts/`; the plan is saved next to them.

| Start (4 runs each) | Self-EOS | Ran to 8192 cap | Loop flag | Median length | Median distinct-bigram |
|---|---|---|---|---|---|
| empty | 4 | 0 | 1 | 1,556 | 0.47 |
| soft | 4 | 0 | **0** | 778 | **0.76** |
| random α=0.25 | 3 | 1 | 2 | 1,351 | 0.35 |
| random α=1 | 1 | **3** | 3 | 8,192 | 0.22 |
| fragment + BOS sink | 4 | 0 | 1 | 1,511 | 0.19 |

## Compared with the 4B

- **Soft is still the healthiest start** (0 loops, most diverse). The self-interpretation behaviour replicates: one run writes garbled Thai inside HTML, then says, in Chinese, *"the message you provided contains many non-standard characters; I'll try to remove them…"*. Others give a coherent Chinese explainer of how to pronounce "Python", a burglary-rate Q&A, and a history multiple-choice question.
- **Random α=1 still loops, but on 8B the loops rarely wake up.** 3/4 ran all 8,192 tokens stuck (`ffff…`, `\.\.\.`). On the 4B, 14/16 looping runs ended themselves (with a 2k cap). The larger model stays stuck longer.
- **The "dream-like middle" did not carry over cleanly.** At α=0.25 the 8B looped in 2/4 runs (`‎…`, `XXXX…ffff`); one run was coherent study notes ("hash functions … I didn't cover them much"). The dial may sit lower on 8B.
- **The empty start is less clean on 8B:** one looped on invisible BOM characters, and others drift into repetition ("what's my name file…", "患者患者…"). On the 4B, empty starts produced clean code.
- **Fragment + sink: repetitive self-questioning** rather than the 4B's coherent documents. Examples: *"How do I process this code to a decision? … Can I process this code to a decision?"*, *"Can you ask me 'Are there any questions unanswered'?"*, and nested `# Make Sure if __name__ == "__main__":`. Diversity is low (median 0.19).
- **The generous budget was barely used.** Only 5/20 runs reached a reduction, and those were mostly the long loops. Runs wake or get stuck well before 2,048 tokens, so memory policy plays little role at this budget.

## Caveats

- n=4 per start; one sampling stream (except empty); unblinded reading.
- 4B and 8B differ in more than size (the lm_head is untied on 8B), and the 4B comparison used a 2k cap and a 512 budget.

## Unresolved

- Is the lower "dream band" on 8B real? An α sweep {0.05, 0.1, 0.25} on 8B would test it (~12 runs).

## Addendum: 8B noise sweep (alpha-sweep-8b-v1)

App `ap-ejuncXgYTv9F7DPps3FZTn`, 12/12 OK. Random KV, evict, same budget; α=0.25 and 1 are reused from the survey (identical config). Texts are in `alpha-sweep-texts/`.

| α | Loop flag | Stuck to 8192 cap | Self-EOS |
|---|---|---|---|
| 0.05 | **4/4** | 1 | 3 |
| 0.1 | 3/4 | 3 | 1 |
| 0.15 | 3/4 | 3 | 1 |
| 0.25 (survey) | 2/4 | 1 | 3 |
| 1.0 (survey) | 3/4 | 3 | 1 |

**No clean "dream band" on the 8B.**
- At every strength tested, random-KV starts mostly fall straight into symbol loops (`➯➯➯⧏⧏⧏`, `ㅜㅜㅜ`, `✕✕✕`, `eeee…ffff`, `XXXX…ffff`), often by token 128. All four α=0.05 seeds even open with the same `⭕`.
- This differs from the 4B, where α=0.1 noise was simply ignored and α 0.25–0.5 gave varied, dream-like text.
- Rare exceptions:
  - α=0.1 seed 47 wrote a short Chinese poem, 《再少年》: *"awakened by the sound of carts, I open the faint sounds, and find only a few lines of poetry…"*;
  - α=0.15 seed 23 burst `$$$$` and then wrote Python.
- **Hypothesis (untested): dilution.** At small α, the 64 near-zero KV slots still absorb attention mass (there is no sink). The model is not "ignoring" them, it is being diluted by them, and the 8B seems more sensitive to this than the 4B.
- **Practical conclusion:** across both sizes, the **soft start** (random embeddings passed through the model) is the robust way to make these models dream: 0 loops on 4B (16 runs) and on 8B (4 runs). Random-KV injection is fragile and size-dependent.
