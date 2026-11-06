"""
Policy loss for GRPO on pre-generated rollouts, with the behaviour policy's log-probs supplied
as `rollout_logprobs`. In the replay setup these are the reference worker's log-probs, copied
in by ReplayTrainer.train_critic_and_policy (same FSDP/Ulysses stack as the policy, so the
ratio is exactly 1 at step 1).

Per token t (loss_mask == 1):

    r_t   = exp(log pi_theta(t) - log pi_ref(t))                     importance ratio
    w_t   = r_t  if 1 - eps_clip_low < r_t < 1 + eps_clip_high else 0   banded, stop-gradient
    pg_t  = - w_t * A_t * log pi_theta(t)                             (same form as SkyRL's rollout_is)

    loss  = reduce_loss(pg, loss_mask)

The KL to the reference is SkyRL's built-in use_kl_loss term (kl_loss_coef * mean_seq(masked_mean(k3)),
added in PolicyWorkerBase._forward_backward_micro from base_action_log_probs), so this loss does
not add one of its own when use_kl_loss is true; it only reports it as a metric. With
use_kl_loss=false (no ref worker, rollout_logprobs from a parquet column) the KL is added here.

Config fields read (all existing AlgorithmConfig fields):
    eps_clip_low, eps_clip_high   band around the behaviour policy (default 0.2 / 0.2)
    use_kl_loss                   true: KL added by SkyRL; false: added here
    kl_loss_coef, kl_estimator_type

Select with:
    trainer.algorithm.policy_loss_type=replay_offline
    trainer.algorithm.use_kl_loss=true          # ref worker on REF_NODES>=1 (or colocate_policy_ref)
The module must be imported in the driver before Ray is initialised (main_replay.py does this);
SkyRL then syncs the registry to the workers.
"""
from typing import Optional, Tuple

import torch

from skyrl.backends.skyrl_train.utils.ppo_utils import compute_approx_kl, reduce_loss, register_policy_loss
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean

LOSS_NAME = "replay_offline"


@register_policy_loss(LOSS_NAME)
def replay_offline_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: Optional[torch.Tensor],
    advantages: torch.Tensor,
    config,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, dict]:
    assert rollout_logprobs is not None, (
        f"{LOSS_NAME} needs rollout_logprobs (the reference worker's log-probs; see replay/main_replay.py)"
    )
    if loss_mask is None:
        loss_mask = torch.ones_like(log_probs)
    loss_mask = loss_mask.to(log_probs.dtype)

    # importance ratio against the behaviour policy, banded (stop-gradient weight)
    delta = (log_probs - rollout_logprobs).clamp(-20.0, 20.0)
    ratio = torch.exp(delta)
    in_band = (ratio > 1.0 - config.eps_clip_low) & (ratio < 1.0 + config.eps_clip_high)
    weight = torch.where(in_band, ratio, torch.zeros_like(ratio)).detach()
    pg_loss = -(weight * advantages * log_probs)
    pg_loss = reduce_loss(pg_loss, loss_mask)

    # KL to the behaviour policy, normalised like SkyRL's built-in use_kl_loss path
    kl_tok = compute_approx_kl(
        log_probs, rollout_logprobs, loss_mask=loss_mask,
        kl_estimator_type=getattr(config, "kl_estimator_type", "k3"),
    )
    kl_loss = masked_mean(kl_tok, loss_mask, dim=-1).mean()
    kl_coef = float(getattr(config, "kl_loss_coef", 0.0))
    use_builtin_kl = bool(getattr(config, "use_kl_loss", False))

    loss = pg_loss if use_builtin_kl else pg_loss + kl_coef * kl_loss

    with torch.no_grad():
        n = loss_mask.sum()
        if n.item() == 0:
            # Padding micro-batch (SkyRL pads ranks with fewer micro-batches; loss_mask is all zero).
            # SkyRL averages metrics over micro-batches, so report neutral values here instead of
            # zeros that would pull ratio_mean towards 0 and clip_ratio towards 0.
            metrics = {"clip_ratio": 0.0, "replay/ratio_mean": 1.0, "replay/ratio_max": 1.0,
                       "replay/kl_loss": 0.0, "replay/pg_loss": 0.0}
        else:
            metrics = {
                "clip_ratio": ((~in_band).to(log_probs.dtype) * loss_mask).sum().item() / n.item(),  # fraction out of band
                "replay/ratio_mean": (ratio * loss_mask).sum().item() / n.item(),
                "replay/ratio_max": (ratio * loss_mask).max().item(),
                "replay/kl_loss": kl_loss.item(),   # metric only when use_kl_loss is true; SkyRL's policy_kl is the applied term
                "replay/pg_loss": pg_loss.item(),
            }
    return loss, metrics