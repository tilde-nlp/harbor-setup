#!/usr/bin/env python
"""
Numerical test of SkyRL's loss functions with a partial loss mask. CPU only, no Ray, no GPU.
Run inside the container, where `skyrl` is installed:

    python -m replay.test_loss_masking

What it proves, using the installed SkyRL code (not a copy of it):
  1. ppo_policy_loss: changing the policy log-probs at loss_mask == 0 positions does not change
     the loss, and the gradient of the loss w.r.t. those log-probs is exactly 0.
  2. compute_approx_kl + masked_mean (the use_kl_loss path in PolicyWorkerBase._forward_backward_micro):
     same two properties.
  3. masked_mean over entropy: same.
  4. Advantages at masked positions may be non-zero (they are multiplied by response_mask, not
     loss_mask) and that is harmless because of 1.

Call signatures differ between SkyRL versions; the helpers below introspect the signature and
pass only the arguments it has. If something cannot be constructed, the error shows the
signature so the call can be adjusted.
"""
import inspect
import sys

import torch

torch.manual_seed(0)


def call_with_known_args(fn, **kwargs):
    params = inspect.signature(fn).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    missing = [p.name for p in params.values()
               if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
               and p.name not in accepted]
    if missing:
        raise TypeError(f"{fn.__name__}{inspect.signature(fn)} needs {missing}; provided {list(kwargs)}")
    return fn(**accepted)


def get_algorithm_config():
    from skyrl.train.config import SkyRLTrainConfig
    cfg = SkyRLTrainConfig()
    alg = cfg.trainer.algorithm
    # plain PPO loss, no off-policy correction, token-level reduction
    for name, val in (("policy_loss_type", "regular"), ("loss_reduction", "token_mean")):
        if hasattr(alg, name):
            setattr(alg, name, val)
    return alg


def make_batch(B=3, R=12):
    log_probs = (-torch.rand(B, R) * 3).requires_grad_(True)
    old_log_probs = (log_probs.detach() + 0.1 * torch.randn(B, R))
    ref_log_probs = (log_probs.detach() + 0.3 * torch.randn(B, R))
    loss_mask = torch.ones(B, R)
    loss_mask[0, 3:7] = 0       # a tool-output block in the middle
    loss_mask[1, :2] = 0        # leading padding
    loss_mask[2, 8:] = 0        # trailing block
    advantages = torch.randn(B, 1).expand(B, R).clone()   # GRPO: one scalar per sequence, on ALL tokens
    return log_probs, old_log_probs, ref_log_probs, advantages, loss_mask


def perturbed(x, mask, delta=5.0):
    """Copy of x with a large change at mask == 0 positions."""
    y = x.detach().clone()
    y[mask == 0] += delta
    return y.requires_grad_(True)


def check(name, loss_fn, log_probs, loss_mask):
    """loss_fn(log_probs) -> scalar. Verify invariance and zero gradient on masked positions."""
    l1 = loss_fn(log_probs)
    l2 = loss_fn(perturbed(log_probs, loss_mask))
    l1.backward()
    g = log_probs.grad
    inv = torch.allclose(l1.detach(), l2.detach(), rtol=1e-5, atol=1e-6)
    zero_grad = bool((g[loss_mask == 0] == 0).all())
    nonzero_grad = bool((g[loss_mask == 1] != 0).any())
    status = "OK  " if (inv and zero_grad and nonzero_grad) else "FAIL"
    print(f"{status} {name:28s} loss={l1.item():+.6f} perturbed={l2.item():+.6f} "
          f"invariant={inv} grad_on_masked_zero={zero_grad} grad_on_kept_nonzero={nonzero_grad}")
    log_probs.grad = None
    return inv and zero_grad and nonzero_grad


def main():
    from skyrl.backends.skyrl_train.utils import ppo_utils
    from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean

    alg = get_algorithm_config()
    log_probs, old_log_probs, ref_log_probs, advantages, loss_mask = make_batch()
    print(f"loss_mask:\n{loss_mask.int()}\n")
    results = []

    # 1. policy loss (the registered loss for policy_loss_type="regular")
    try:
        from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
        policy_loss_fn = PolicyLossRegistry.get(alg.policy_loss_type)
    except Exception:
        policy_loss_fn = ppo_utils.ppo_policy_loss

    def policy_loss(lp):
        out = call_with_known_args(
            policy_loss_fn, log_probs=lp, old_log_probs=old_log_probs, advantages=advantages,
            config=alg, loss_mask=loss_mask, rollout_logprobs=None, rollout_action_logprobs=None,
        )
        return out[0] if isinstance(out, tuple) else out

    results.append(check(f"policy loss ({policy_loss_fn.__name__})", policy_loss, log_probs, loss_mask))

    # 2. KL loss, as in PolicyWorkerBase._forward_backward_micro
    def kl_loss(lp):
        kl = call_with_known_args(
            ppo_utils.compute_approx_kl, log_probs=lp, log_probs_base=ref_log_probs,
            loss_mask=loss_mask, kl_estimator_type=getattr(alg, "kl_estimator_type", "k3"),
        )
        return masked_mean(kl, loss_mask, dim=-1).mean()

    results.append(check("kl loss (compute_approx_kl)", kl_loss, log_probs, loss_mask))

    # 3. entropy-style masked mean
    results.append(check("masked_mean", lambda lp: masked_mean(-lp, loss_mask), log_probs, loss_mask))

    # 4. advantage at masked positions is non-zero and that does not matter (covered by 1)
    print(f"\nadvantage at masked positions (non-zero by design): "
          f"{advantages[loss_mask == 0][:4].tolist()}")
    ok = all(results)
    print("\nALL OK: masked tokens contribute neither loss nor gradient" if ok else "\nFAILURES above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()