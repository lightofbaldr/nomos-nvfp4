"""Token sampling for Gemma-4 generation.

Three things happen on the raw logits before we pick a token:

  1. final_logit_softcap(logits, cap=30.0)
     Gemma-4's final_logit_softcapping: logits = cap * tanh(logits / cap).
     Keeps the distribution well-behaved without hard clipping.

  2. apply_rep_penalty(logits, gen_tokens, rep_penalty=1.3, window=64)
     Windowed repetition penalty. For each token in the last `window`
     generated tokens, divide its positive logit by rep_penalty (or
     multiply its negative logit by rep_penalty). A full-history penalty
     starves common function words after a couple of sentences, so we
     cap the window.

  3. mask_eos_if_too_early(logits, gen_count, min_gen=4)
     The Opus distill model loves to emit <turn|> (token 106) or eos (1)
     at position 0 of generation. This forces those down to -inf for the
     first `min_gen` generated tokens so the model has to actually say
     something before it's allowed to end.

Then the sampling strategy:

  sample_top_p(logits, temperature, top_p, max_keep, rng_state)
    Temperature-softmaxed top-p (nucleus) sampling. Partial top-k selection
    capped at `max_keep` candidates for bounded work (important since the
    Gemma-4 vocab is 262144). Returns the picked token id and the updated
    rng_state (LCG inverse-CDF sampler).
"""

from std.math import exp, tanh
from std.collections import List


def final_logit_softcap(mut logits: List[Float32], vocab: Int, cap: Float32 = 30.0):
    """In-place: logits = cap * tanh(logits / cap).

    Expanded as `cap * (e^{2v} - 1) / (e^{2v} + 1)` with v clamped to
    [-10, 10] to prevent overflow — the equivalent of Python's numerically
    stable tanh that we already use in gemma4_ops.gelu.  A non-positive cap
    disables softcapping (Qwen profile contract); do not divide by zero and
    turn ordinary logits into zeros/NaNs.
    """
    if cap <= Float32(0.0):
        return
    for i in range(vocab):
        var v = logits[i] / cap
        if v > 10.0: v = 10.0
        if v < -10.0: v = -10.0
        var e2v = exp(2.0 * v)
        logits[i] = cap * ((e2v - 1.0) / (e2v + 1.0))


def apply_rep_penalty(mut logits: List[Float32], gen_tokens: List[Int],
                      vocab: Int, rep_penalty: Float32 = 1.3, window: Int = 64):
    """Windowed repetition penalty on the last `window` tokens of gen_tokens.

    Matches the HF `RepetitionPenaltyLogitsProcessor` convention: for each
    token we've recently generated, divide its positive logit by the penalty
    or multiply its negative logit by the penalty. Pulls the logit toward
    zero either way, making the token less attractive.
    """
    if len(gen_tokens) == 0: return
    var start = 0
    if len(gen_tokens) > window: start = len(gen_tokens) - window
    # Apply penalty ONCE per unique token in window (HF semantics). The
    # previous code applied it once per OCCURRENCE; over a softcapped
    # (-30, 30) logit range that compounded as penalty^N per frequent
    # token, crushing common code/prose tokens below the long-tail emoji
    # and unused-id distribution. Greedy then deterministically marched
    # into those tails (emoji spam / hedge-restart). Bug B, 2026-05-29.
    var seen = List[Bool](capacity=vocab)
    for _ in range(vocab): seen.append(False)
    for gi in range(start, len(gen_tokens)):
        var gt = gen_tokens[gi]
        if gt >= 0 and gt < vocab and not seen[gt]:
            seen[gt] = True
            # Don't penalize EOS-class / channel tokens — penalizing them
            # produces runaway tails when the model wants to wrap up.
            if gt == 1 or gt == 106 or gt == 101: continue
            if logits[gt] > 0.0:
                logits[gt] /= rep_penalty
            else:
                logits[gt] *= rep_penalty


def mask_special_tokens(mut logits: List[Float32], vocab: Int):
    """Force genuinely-unused token logits to -inf so the model can never
    emit them as visible text. PRESERVES structural special tokens that
    the chat template + tool grammar rely on.

    Gemma-4-31B vocab (probed via tokenizer-svc, md5 72b1044…):

        id 0           = <pad>             ← masked
        id 1           = <eos>             ← SKIPPED (mask_eos_if_too_early handles)
        id 2           = <bos>             ← masked
        id 3           = <unk>             ← masked
        id 4           = <mask>            ← masked
        id 5           = [multimodal]      ← masked
        id 6..45       = <unused>          ← masked
        id 46          = <|tool>           ← KEPT (tool declaration)
        id 47          = <tool|>           ← KEPT
        id 48          = <|tool_call>      ← KEPT (tool call opener — critical for tool calling)
        id 49          = <tool_call|>      ← KEPT
        id 50          = <|tool_response>  ← KEPT
        id 51          = <tool_response|>  ← KEPT
        id 52          = <|"|>             ← KEPT (string delim inside tool args)
        id 53..97      = <unused>          ← masked
        id 98          = <|think|>         ← KEPT (thinking toggle)
        id 99          = <unused>          ← masked
        id 100         = <|channel>        ← KEPT (thought channel open)
        id 101         = <channel|>        ← KEPT (thought channel close — engine line 815 reads this!)
        id 102..104    = <unused>          ← masked
        id 105         = <|turn>           ← KEPT (turn marker)
        id 106         = <turn|>           ← SKIPPED (mask_eos_if_too_early handles)
        id 107..255998 = normal vocab
        id 255999      = empty/special    ← masked
        id 256000      = empty/special    ← masked
        id 256001..    = <unused89> upward ← masked

    2026-05-30 fix: the prior `for i in range(6, 106)` blanket masked the 11
    structural specials inside that range, blocking the base model from ever
    emitting <|tool_call>=48 et al. This regression made the model fall back
    to plaintext '<call:read_file{...}>' instead of the native tool-call
    token sequence. Restoring the allowlist below restores token 48 emission.
    """
    logits[0] = Float32(-1e9)
    logits[2] = Float32(-1e9)
    logits[3] = Float32(-1e9)
    logits[4] = Float32(-1e9)
    logits[5] = Float32(-1e9)
    for i in range(6, 106):
        if (i == 46 or i == 47 or i == 48 or i == 49 or i == 50 or i == 51
            or i == 52 or i == 98 or i == 100 or i == 101 or i == 105):
            continue
        logits[i] = Float32(-1e9)
    logits[255999] = Float32(-1e9)
    logits[256000] = Float32(-1e9)
    for i in range(256001, vocab):
        logits[i] = Float32(-1e9)


def mask_eos_if_too_early(mut logits: List[Float32], gen_count: Int, min_gen: Int = 4):
    """Force EOS/end-of-turn logits to -inf for the first `min_gen` generated tokens.

    Gemma-4 Opus distill model has a strong tendency to predict <turn|> (106)
    or eos_token_id (1) in the very first generation step, producing empty
    or one-token responses. `min_gen` gives it a floor — must generate at
    least that many real tokens before ending is allowed.
    """
    if gen_count < min_gen:
        logits[1] = -1e9
        logits[106] = -1e9


def fix_contraction_me_bug(mut logits: List[Float32], last_token: Int):
    """After an "-n't" contraction stem like ` isn`/`wasn`/`aren`/..., the base
    `unsloth/gemma-4-31b-it` model's distribution over the next token is
    spread across many wrong alternatives: ` me` (786), ` as` (527), ` a`
    (a lot), and only occasionally `'` (236789, the byte-level apostrophe).
    In the stem→? context, `'` is almost always the right answer in English —
    anything else is wrong.

    Fix: force `'` (236789) to win by lifting its logit 50 above the max.
    Works together with `fix_contraction_t_bug` which does the same for `t`
    (236745) in the second step. The net effect is: once the model types
    ` isn`, we force it to complete as ` isn't` regardless of what else it
    wanted to say.

    Stems (tokenized with leading space and without — mid-word tokenization
    happens in the no-space form):
      ` isn`, ` wasn`, ` aren`, ` doesn`, ` don`, ` didn`, ` hasn`, ` haven`,
      ` couldn`, ` shouldn`, ` wouldn`, ` won`, ` weren`, ` hadn`
      plus their no-leading-space variants where they exist as single tokens.
    """
    var is_stem = False
    # Merged stems with leading space
    if last_token == 5889:   is_stem = True  # ' isn'
    elif last_token == 7289:   is_stem = True  # ' wasn'
    elif last_token == 10050:  is_stem = True  # ' aren'
    elif last_token == 4038:   is_stem = True  # ' doesn'
    elif last_token == 1537:   is_stem = True  # ' don'
    elif last_token == 3782:   is_stem = True  # ' didn'
    elif last_token == 18116:  is_stem = True  # ' hasn'
    elif last_token == 10366:  is_stem = True  # ' haven'
    elif last_token == 9225:   is_stem = True  # ' couldn'
    elif last_token == 18461:  is_stem = True  # ' shouldn'
    elif last_token == 10369:  is_stem = True  # ' wouldn'
    elif last_token == 2810:   is_stem = True  # ' won'
    elif last_token == 19310:  is_stem = True  # ' weren'
    elif last_token == 26180:  is_stem = True  # ' hadn'
    # Merged stems without leading space (mid-word or start-of-generation)
    elif last_token == 133792: is_stem = True  # 'isn'
    elif last_token == 5956:   is_stem = True  # 'aren'
    elif last_token == 106908: is_stem = True  # 'doesn'
    elif last_token == 13246:  is_stem = True  # 'don'
    elif last_token == 136150: is_stem = True  # 'didn'
    elif last_token == 57260:  is_stem = True  # 'haven'
    elif last_token == 41979:  is_stem = True  # 'won'
    if not is_stem: return

    # Lift `'` (236789) well above the current max so it wins after softmax.
    var mx: Float32 = logits[0]
    for i in range(1, 262144):
        if logits[i] > mx: mx = logits[i]
    logits[236789] = mx + 50.0


def fix_contraction_t_bug(mut logits: List[Float32], last_token: Int, prev_token: Int):
    """Second-step fix: after a stem + `'` pair (e.g. ` isn` `'`), the model
    sometimes inserts markdown noise (`*`, `**`, `//`) before picking `t`.
    When we detect that pattern (prev was a stem, last was the `'` apostrophe
    at token 236789), forbid the common markdown/comment tokens so `t` (236745)
    wins cleanly. Same as fix_contraction_me_bug, this has no effect outside
    the stem→'→t context.
    """
    if last_token != 236789: return  # not the apostrophe
    var is_stem = False
    if prev_token == 5889:   is_stem = True  # ' isn'
    elif prev_token == 7289:   is_stem = True  # ' wasn'
    elif prev_token == 10050:  is_stem = True  # ' aren'
    elif prev_token == 4038:   is_stem = True  # ' doesn'
    elif prev_token == 1537:   is_stem = True  # ' don'
    elif prev_token == 3782:   is_stem = True  # ' didn'
    elif prev_token == 18116:  is_stem = True  # ' hasn'
    elif prev_token == 10366:  is_stem = True  # ' haven'
    elif prev_token == 9225:   is_stem = True  # ' couldn'
    elif prev_token == 18461:  is_stem = True  # ' shouldn'
    elif prev_token == 10369:  is_stem = True  # ' wouldn'
    elif prev_token == 2810:   is_stem = True  # ' won'
    elif prev_token == 19310:  is_stem = True  # ' weren'
    elif prev_token == 26180:  is_stem = True  # ' hadn'
    elif prev_token == 133792: is_stem = True  # 'isn'
    elif prev_token == 5956:   is_stem = True  # 'aren'
    elif prev_token == 106908: is_stem = True  # 'doesn'
    elif prev_token == 13246:  is_stem = True  # 'don'
    elif prev_token == 136150: is_stem = True  # 'didn'
    elif prev_token == 57260:  is_stem = True  # 'haven'
    elif prev_token == 41979:  is_stem = True  # 'won'
    if not is_stem: return

    # Force `t` (236745) to win by pinning it well above the maximum logit.
    # We tried the softer approach (forbidding markdown/comment distractors)
    # and the model just picked other garbage (double apostrophe, `'til`,
    # escape chars). In the specific stem→'→? context, `t` is the only
    # correct continuation in English — anything else is wrong. Lifting
    # `t` 50 above max guarantees it wins after temp+softmax while leaving
    # the rest of the distribution intact for downstream logit bookkeeping.
    var mx: Float32 = logits[0]
    for i in range(1, 262144):
        if logits[i] > mx: mx = logits[i]
    logits[236745] = mx + 50.0


struct SampleResult:
    """Token pick + updated rng_state. Returned as a struct so the caller
    only has to thread rng_state through generate() once."""
    var token_id: Int
    var rng_state: UInt64

    def __init__(out self, token_id: Int, rng_state: UInt64):
        self.token_id = token_id
        self.rng_state = rng_state


def argmax(logits: List[Float32], vocab: Int) -> Int:
    """Greedy: pick the single highest-logit token."""
    var best: Float32 = logits[0]
    var bi: Int = 0
    for i in range(1, vocab):
        if logits[i] > best:
            best = logits[i]
            bi = i
    return bi


def sample_top_p(logits: List[Float32], vocab: Int, temperature: Float32,
                 top_p: Float32, max_keep: Int, rng_state: UInt64) -> SampleResult:
    """Temperature-softmaxed top-p (nucleus) sampling with bounded top-k.

    1. Softmax the logits with the given temperature (max-shift for stability).
    2. Pull tokens in descending probability order until cumulative prob >= top_p
       or we've picked `max_keep` (bound on the scan cost — vocab is 262144 so we
       cap selection at 64 by default).
    3. Renormalize the selected nucleus and sample via inverse CDF using an LCG
       on `rng_state`.

    Returns SampleResult(picked_token, advanced_rng_state). Falls back to argmax
    if for some reason no candidates have positive probability.
    """
    # Argmax fast path for greedy decoding. exp((x - max) / temperature) at
    # temperature=0 is 0/0 = NaN, which then propagates through top-p selection
    # and lets garbage sentinel tokens (e.g. <unused45>) win because NaN compares
    # weirdly. The existing argmax fallback below only fires if len(keep_idx)==0,
    # which doesn't reliably happen under NaN. Catch greedy here directly so
    # callers requesting temperature=0 get clean argmax output.
    if temperature < Float32(0.01):
        return SampleResult(argmax(logits, vocab), rng_state)

    # Temperature softmax (full vocab, max-shifted)
    var mx: Float32 = -1e9
    for i in range(vocab):
        if logits[i] > mx: mx = logits[i]
    var sum_e: Float32 = 0.0
    var probs = List[Float32](capacity=vocab)
    for i in range(vocab):
        var p = exp((logits[i] - mx) / temperature)
        probs.append(p)
        sum_e += p
    if sum_e > 0.0:
        for i in range(vocab): probs[i] /= sum_e

    # Top-p nucleus selection — repeated best-scan up to max_keep
    var keep_idx = List[Int](capacity=max_keep)
    var keep_val = List[Float32](capacity=max_keep)
    var cum: Float32 = 0.0
    var used = List[Bool](capacity=vocab)
    for _ in range(vocab): used.append(False)
    for _ in range(max_keep):
        var bv: Float32 = -1.0
        var bj: Int = -1
        for i in range(vocab):
            if not used[i] and probs[i] > bv:
                bv = probs[i]
                bj = i
        if bj < 0 or bv <= 0.0: break
        used[bj] = True
        keep_idx.append(bj)
        keep_val.append(bv)
        cum += bv
        if cum >= top_p: break

    if len(keep_idx) == 0:
        return SampleResult(argmax(logits, vocab), rng_state)

    # Renormalize selected nucleus
    var tot: Float32 = 0.0
    for i in range(len(keep_idx)): tot += keep_val[i]
    for i in range(len(keep_idx)): keep_val[i] /= tot

    # LCG advance + inverse CDF
    var new_state = (rng_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    var r = Float32(new_state >> 33) / Float32(UInt64(1) << 31)
    var acc: Float32 = 0.0
    var bi = keep_idx[0]
    for i in range(len(keep_idx)):
        acc += keep_val[i]
        if r < acc:
            bi = keep_idx[i]
            break
    return SampleResult(bi, new_state)


def mask_tool_grammar(mut logits: List[Float32], gen_tokens: List[Int], vocab: Int):
    """Stage-A grammar constraint for Gemma-4 tool calls.

    Operates purely on structural special-token IDs (no detokenization needed —
    every structural piece is a single token in Gemma-4's vocab):
        48  <|tool_call>   49  <tool_call|>   52  <|"|>
        236782 {   236783 }   236787 :        1 <eos>   106 <turn|>

    The Pi baseline showed the *content* the base produces is correct, but the
    unconstrained sampler emits malformed tool-call SYNTAX (unterminated <|"|>
    strings, premature <tool_call|> before the object closes, empty objects),
    which the OpenAI-compat parser can't recover → tool call dropped → agent loop
    stalls. This masks the few structural tokens that would create those
    malformations, so the model physically cannot emit them. It only ever fires
    while INSIDE an open tool call; normal thought/content generation is
    untouched. String/number/bool VALUE content is left free — we only constrain
    the structural skeleton (Stage B will add schema-driven required-key checks).
    """
    alias OPEN = 48
    alias CLOSE = 49
    alias QUOTE = 52
    alias LBRACE = 236782
    alias RBRACE = 236783
    alias COLON = 236787
    alias EOS = 1
    alias TURN = 106
    alias NEG = Float32(-1e9)

    var n = len(gen_tokens)
    # Find the last <|tool_call> with no matching <tool_call|> after it.
    var open_idx = -1
    for i in range(n):
        var t = gen_tokens[i]
        if t == OPEN:
            open_idx = i
        elif t == CLOSE:
            open_idx = -1
    if open_idx < 0:
        return  # not inside a tool call — no constraint

    # Tally structural tokens emitted since the open.
    var quote_count = 0
    var lbrace = 0
    var rbrace = 0
    var colon = 0
    for i in range(open_idx + 1, n):
        var t = gen_tokens[i]
        if t == QUOTE:
            quote_count += 1
        elif t == LBRACE:
            lbrace += 1
        elif t == RBRACE:
            rbrace += 1
        elif t == COLON:
            colon += 1
    var str_open = (quote_count % 2) == 1
    var brace_balanced = (lbrace > 0) and (lbrace == rbrace)
    var brace_open = lbrace > rbrace

    # R4: never open a nested tool call while already inside one.
    logits[OPEN] = NEG

    # R1: inside an unterminated <|"|> string → cannot end the string-bearing
    # structures. Forces the model to close the string first. (Fixes the
    # `content:<|"|>...== "FizzBu`-style truncations from the baseline.)
    if str_open:
        logits[CLOSE] = NEG
        logits[EOS] = NEG
        logits[TURN] = NEG
        logits[RBRACE] = NEG
        return

    # R2: the object brace must be opened and closed before the call can finish.
    # (Fixes premature <tool_call|>/<turn|> before `}`.)
    if not brace_balanced:
        logits[CLOSE] = NEG
        logits[EOS] = NEG
        logits[TURN] = NEG

    # R3: object must be non-empty — forbid `}` until at least one `key:` colon
    # has appeared. (Fixes empty `bash{}`-style calls.)
    if brace_open and colon == 0:
        logits[RBRACE] = NEG
