"""
Checks on the pregenerated-rollout parquet, before a training job reads it.

Columns (skyrl_extensions/pregenerated/generator.py): uid, rep, prompt, prompt_ids, response_ids,
loss_mask, score, stop_reason. Hard violations FAIL; statistics that make the gradient lopsided WARN.
"""
import json
import os

import numpy as np

from checks.lib import FAIL, PASS, SKIP, WARN, Result, check

SCORE_MAX = 10.0             # skyrl_extensions/pregenerated/generator.py
MAX_ZERO_ADV_GROUPS = 0.5    # WARN: share of groups whose scores are all equal (no gradient from them)
MAX_ADV_IMBALANCE = 0.2      # WARN: |sum(A*tokens)| / sum(|A|*tokens); measured on the so_openq data: 0.07
REQUIRED = ["uid", "rep", "prompt", "prompt_ids", "response_ids", "loss_mask", "score", "stop_reason"]


def _lists(col):
    """Arrow list column -> (lengths, flat values) as numpy."""
    import pyarrow as pa
    import pyarrow.compute as pc

    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    return pc.list_value_length(arr).to_numpy(zero_copy_only=False).astype(np.int64), pc.list_flatten(arr).to_numpy(zero_copy_only=False)


def validate_pregen(path, n_samples, max_seq_len=None, vocab_size=None, eos_id=None, assistant_id=None):
    """-> (errors, warnings, stats). Pure function of the file: used by the check and by the tests."""
    import pyarrow.parquet as pq

    errors, warnings, stats = [], [], {}
    t = pq.read_table(path)
    missing = [c for c in REQUIRED if c not in t.column_names]
    if missing:
        return ["missing column(s) {}".format(missing)], warnings, stats
    uid = np.array(t["uid"].to_pylist())
    rep = t["rep"].to_numpy().astype(np.int64)
    score = t["score"].to_numpy().astype(np.float64)
    stop = np.array(t["stop_reason"].to_pylist())
    n_rows = len(uid)
    stats.update(rows=n_rows)
    for c in ("prompt_ids", "response_ids", "loss_mask"):
        if t[c].null_count:
            errors.append("{}: {} null rows".format(c, t[c].null_count))
    if errors:
        return errors, warnings, stats
    p_len, p_val = _lists(t["prompt_ids"])
    r_len, r_val = _lists(t["response_ids"])
    m_len, m_val = _lists(t["loss_mask"])

    # groups
    uniq, inv, counts = np.unique(uid, return_inverse=True, return_counts=True)
    stats.update(prompts=len(uniq))
    bad = uniq[counts != n_samples]
    if len(bad):
        errors.append("{} prompts without exactly {} rollouts (n_samples_per_prompt), e.g. {}".format(len(bad), n_samples, bad[:3].tolist()))
    if ((rep < 0) | (rep >= n_samples)).any():
        errors.append("rep outside 0..{} in {} rows".format(n_samples - 1, int(((rep < 0) | (rep >= n_samples)).sum())))
    if len(set(zip(uid.tolist(), rep.tolist()))) != n_rows:
        errors.append("duplicate (uid, rep) rows")

    # per-token columns
    if (m_len != r_len).any():
        errors.append("{} rows where len(loss_mask) != len(response_ids)".format(int((m_len != r_len).sum())))
        return errors, warnings, stats
    if (r_len == 0).any() or (p_len == 0).any():
        errors.append("{} rows with an empty prompt or response".format(int(((r_len == 0) | (p_len == 0)).sum())))
        return errors, warnings, stats
    if not np.isin(m_val, (0, 1)).all():
        errors.append("loss_mask has values other than 0/1")
    r_end = np.cumsum(r_len)
    r_start = r_end - r_len
    mask_sum = np.add.reduceat(m_val.astype(np.int64), r_start)
    if (mask_sum == 0).any():
        errors.append("{} rows with no loss token (loss_mask all zero)".format(int((mask_sum == 0).sum())))
    longest = int((p_len + r_len).max())
    stats.update(longest=longest, tokens=int((p_len + r_len).sum()), loss_tokens=int(mask_sum.sum()),
                 masked_share=float(1 - mask_sum.sum() / r_len.sum()))
    if max_seq_len is not None and longest > max_seq_len:
        errors.append("longest prompt + response is {} tokens, above max_seq_len={}".format(longest, max_seq_len))
    if vocab_size is not None:
        top = int(max(p_val.max(), r_val.max()))
        if top >= vocab_size or min(p_val.min(), r_val.min()) < 0:
            errors.append("token id {} outside the vocabulary (size {})".format(top, vocab_size))

    # sequence ends
    if assistant_id is not None:
        p_last = p_val[np.cumsum(p_len) - 1]
        if (p_last != assistant_id).any():
            errors.append("{} prompts do not end with the generation-prompt token {}".format(int((p_last != assistant_id).sum()), assistant_id))
    if eos_id is not None:
        r_last = r_val[r_end - 1]
        is_stop = stop == "stop"
        n_bad = int((is_stop & (r_last != eos_id)).sum())
        if n_bad:
            errors.append("{} rows with stop_reason=stop do not end with eos {}".format(n_bad, eos_id))
        last_masked = m_val[r_end - 1] == 0
        if (is_stop & last_masked).any():
            errors.append("{} finished rows have loss_mask 0 on the final eos (the model would not learn to stop)".format(int((is_stop & last_masked).sum())))
        stats.update(stop_share=float(is_stop.mean()))

    # rewards and advantages (Dr. GRPO: score - group mean, as the trainer computes them)
    if (score < 0).any() or (score > SCORE_MAX).any() or np.isnan(score).any():
        errors.append("score outside 0..{} or NaN".format(SCORE_MAX))
    g_mean = np.bincount(inv, weights=score) / counts
    adv = (score - g_mean[inv]) / SCORE_MAX
    g_spread = np.bincount(inv, weights=np.abs(adv))
    zero_share = float((g_spread == 0).mean())
    den = float((np.abs(adv) * mask_sum).sum())
    imbalance = float((adv * mask_sum).sum() / den) if den else 0.0
    stats.update(zero_advantage_groups=zero_share, advantage_token_imbalance=imbalance, mean_score=float(score.mean()))
    if zero_share == 1.0:
        errors.append("every group has all-equal scores: no advantage, no gradient")
    elif zero_share > MAX_ZERO_ADV_GROUPS:
        warnings.append("{:.0%} of the groups have all-equal scores (they give no gradient)".format(zero_share))
    if abs(imbalance) > MAX_ADV_IMBALANCE:
        warnings.append("token-weighted advantages are lopsided: {:+.2f} (0 = balanced). With token_mean the batch gradient "
                        "mostly {} log-probs".format(imbalance, "raises" if imbalance > 0 else "lowers"))
    return errors, warnings, stats


def is_pregen(ctx):
    return "pregenerated" in str(ctx.get("entrypoint.module", ""))


@check("pre", "data.pregen")
def data_pregen(ctx):
    """Pregenerated parquet: group sizes, mask/response lengths, sequence ends, vocabulary, score range, advantage balance."""
    if not is_pregen(ctx):
        return Result(SKIP, "not a pregenerated-rollout config")
    path = (ctx.get("skyrl.data.train_data") or [None])[0]
    if not path or not os.path.exists(path):
        return Result(FAIL, "skyrl.data.train_data[0] does not exist: {}".format(path))
    cfg = json.load(open(os.path.join(ctx.model_dir, "config.json")))
    assistant_id = None
    golden = os.path.join(ctx.model_dir, "golden_tokenization.jsonl")
    if os.path.exists(golden):  # the generation-prompt token is whatever the golden prompts end with
        assistant_id = json.loads(open(golden).readline())["ids"][-1]
    errors, warnings, stats = validate_pregen(
        path, ctx.get("skyrl.generator.n_samples_per_prompt"), ctx.get("skyrl.trainer.algorithm.max_seq_len"),
        cfg.get("vocab_size"), cfg.get("eos_token_id"), assistant_id)
    line = "{rows} rollouts / {prompts} prompts, longest {longest} tokens, {masked_share:.1%} of response tokens masked, " \
           "advantage balance {advantage_token_imbalance:+.2f}".format(**stats) if "advantage_token_imbalance" in stats else path
    if errors:
        return Result(FAIL, "{} problem(s) in {}".format(len(errors), os.path.basename(path)), errors + warnings, stats)
    if warnings:
        return Result(WARN, line, warnings, stats)
    return Result(PASS, line, data=stats)
