from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv


def masked_eta_losses(
    eta_mu: torch.Tensor,
    eta_log_var: torch.Tensor | None,
    eta_tgt: torch.Tensor,
    eta_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    eta_mu: [B, R, O]
    eta_log_var: [B, R, O] or None
    eta_tgt:  [B, R, O]
    eta_mask: [B, R, O] in {0,1}
    returns: (loss, mae, rmse, nll)
    """
    m = (eta_mask > 0.5)
    if int(m.sum().item()) == 0:
        z = eta_mu.sum() * 0.0
        return z, z, z, z

    diff = eta_mu[m] - eta_tgt[m]
    if eta_log_var is not None:
        lv = torch.clamp(eta_log_var[m], min=-10.0, max=10.0)
        inv_var = torch.exp(-lv)
        nll = 0.5 * ((diff * diff) * inv_var + lv)
        loss = nll.mean()
    else:
        # fallback for legacy point-estimate heads
        nll = 0.5 * (diff * diff)
        loss = F.smooth_l1_loss(eta_mu[m], eta_tgt[m], reduction="mean")
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff * diff).mean())
    return loss, mae, rmse, nll.mean()


class JointMaskablePPO(MaskablePPO):
    """
    Phase 2: PPO loss + eta auxiliary loss (backprop into eta_head)
    Phase 3: inject eta_pred into env BEFORE env.step, so reward uses predicted ETA
    """

    def __init__(self, *args, lambda_eta: float = 0.1, eta_coef: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Backward-compatible alias: eta_coef -> lambda_eta
        if eta_coef is not None:
            lambda_eta = eta_coef
        self.lambda_eta = float(lambda_eta)
        self.eta_coef = float(self.lambda_eta)

    def _inject_eta_pred(self, env: VecEnv) -> None:
        """
        Push current eta_pred (from features_extractor.last_eta) into env.set_eta_pred()
        so that env.step() uses predicted eta.
        """
        fe = getattr(self.policy, "features_extractor", None)
        eta = getattr(fe, "last_eta", None)
        eta_sigma = getattr(fe, "last_eta_sigma", None)
        if eta_sigma is None:
            eta_log_var = getattr(fe, "last_eta_log_var", None)
            if eta_log_var is not None:
                eta_sigma = torch.exp(0.5 * eta_log_var).clamp(min=1e-6)
        if eta is None and eta_sigma is None:
            return

        # NOTE: do NOT detach in Phase 3 injection, env wants numpy anyway (no grad through env)
        eta_np = None if eta is None else eta.detach().cpu().numpy()  # [n_env, R, O] or [R, O]
        eta_sigma_np = None if eta_sigma is None else eta_sigma.detach().cpu().numpy()
        if eta_sigma_np is not None:
            eta_sigma_np = np.maximum(np.nan_to_num(eta_sigma_np, nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6)

        try:
            # DummyVecEnv has .envs list
            if hasattr(env, "envs"):
                for i, w in enumerate(env.envs):
                    # unwrap ActionMasker one level if present
                    e = w
                    while hasattr(e, "env"):
                        e = e.env
                    if hasattr(e, "set_eta_pred"):
                        if eta_np is not None:
                            if eta_np.ndim == 3:
                                e.set_eta_pred(eta_np[i])
                            else:
                                e.set_eta_pred(eta_np)
                    if hasattr(e, "set_eta_sigma"):
                        if eta_sigma_np is not None:
                            if eta_sigma_np.ndim == 3:
                                e.set_eta_sigma(eta_sigma_np[i])
                            else:
                                e.set_eta_sigma(eta_sigma_np)
            else:
                if hasattr(env, "set_eta_pred"):
                    if eta_np is not None:
                        if eta_np.ndim == 3:
                            env.set_eta_pred(eta_np[0])
                        else:
                            env.set_eta_pred(eta_np)
                if hasattr(env, "set_eta_sigma"):
                    if eta_sigma_np is not None:
                        if eta_sigma_np.ndim == 3:
                            env.set_eta_sigma(eta_sigma_np[0])
                        else:
                            env.set_eta_sigma(eta_sigma_np)
        except Exception:
            return

    def collect_rollouts(
        self,
        env: VecEnv,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Copy of MaskablePPO.collect_rollouts with ONE change:
        call self._inject_eta_pred(env) BEFORE env.step(actions)
        """
        assert self._last_obs is not None, "No previous observation. Call reset() first."

        self.policy.set_training_mode(False)
        rollout_buffer.reset()

        if callback is not None:
            callback.on_rollout_start()

        for _ in range(n_rollout_steps):
            if self.use_sde:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                action_masks = get_action_masks(env) if use_masking else None
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions_np = actions.cpu().numpy()

            # ---- Phase 3: inject eta_pred BEFORE env.step ----
            self._inject_eta_pred(env)

            new_obs, rewards, dones, infos = env.step(actions_np)

            self.num_timesteps += env.num_envs

            if callback is not None:
                callback.update_locals(locals())
                if not callback.on_step():
                    return False

            self._update_info_buffer(infos, dones)
            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )

            if use_masking and action_masks is not None and hasattr(rollout_buffer, "action_masks"):
                rollout_buffer.action_masks[rollout_buffer.pos - 1] = action_masks

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            obs_tensor, _ = self.policy.obs_to_tensor(new_obs)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        if callback is not None:
            callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Phase 2: standard PPO train + eta auxiliary loss (from rollout obs eta_tgt/eta_mask)
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        entropy_losses = []
        all_kl_divs = []
        pg_losses, value_losses = [], []
        ppo_losses, eta_losses, total_joint_losses = [], [], []
        eta_maes, eta_rmses, eta_nlls = [], [], []

        continue_training = True

        for _epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                entropy_loss = -log_prob.mean() if entropy is None else -entropy.mean()

                ppo_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss = ppo_loss

                # ----- ETA auxiliary loss (Phase 2) -----
                eta_loss = None
                eta_mae = None
                eta_rmse = None
                eta_nll = None

                obs = rollout_data.observations
                if isinstance(obs, dict) and ("eta_tgt" in obs) and ("eta_mask" in obs):
                    eta_mu = getattr(self.policy.features_extractor, "last_eta", None)
                    eta_log_var = getattr(self.policy.features_extractor, "last_eta_log_var", None)
                    if eta_mu is not None:
                        eta_tgt = obs["eta_tgt"].float()
                        eta_mask = obs["eta_mask"].float()

                        # Ensure [B, R, O]
                        if eta_tgt.dim() == 4:
                            eta_tgt = eta_tgt.squeeze(1)
                        if eta_mask.dim() == 4:
                            eta_mask = eta_mask.squeeze(1)

                        eta_loss, eta_mae, eta_rmse, eta_nll = masked_eta_losses(
                            eta_mu, eta_log_var, eta_tgt, eta_mask
                        )
                        loss = loss + self.lambda_eta * eta_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    approx_kl_divs.append(float(approx_kl_div.detach().cpu().item()))

                pg_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))
                ppo_losses.append(float(ppo_loss.item()))
                total_joint_losses.append(float(loss.item()))

                if eta_loss is not None:
                    eta_losses.append(float(eta_loss.item()))
                if eta_mae is not None:
                    eta_maes.append(float(eta_mae.item()))
                if eta_rmse is not None:
                    eta_rmses.append(float(eta_rmse.item()))
                if eta_nll is not None:
                    eta_nlls.append(float(eta_nll.item()))

            mean_kl = float(np.mean(approx_kl_divs)) if len(approx_kl_divs) > 0 else 0.0
            all_kl_divs.append(mean_kl)

            if self.target_kl is not None and mean_kl > 1.5 * float(self.target_kl):
                continue_training = False
                break

        self._n_updates += self.n_epochs

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)) if pg_losses else 0.0)
        self.logger.record("train/value_loss", float(np.mean(value_losses)) if value_losses else 0.0)
        self.logger.record("train/ppo_loss", float(np.mean(ppo_losses)) if ppo_losses else 0.0)
        self.logger.record("train/approx_kl", float(np.mean(all_kl_divs)) if all_kl_divs else 0.0)
        self.logger.record("train/explained_variance", float(explained_var))
        self.logger.record("train/lambda_eta", float(self.lambda_eta))

        if eta_losses:
            self.logger.record("train/eta_nll_loss", float(np.mean(eta_losses)))
            self.logger.record("train/eta_loss", float(np.mean(eta_losses)))
        if eta_maes:
            self.logger.record("train/eta_mae", float(np.mean(eta_maes)))
        if eta_rmses:
            self.logger.record("train/eta_rmse", float(np.mean(eta_rmses)))
        if eta_nlls:
            self.logger.record("train/eta_nll", float(np.mean(eta_nlls)))
        if total_joint_losses:
            self.logger.record("train/total_joint_loss", float(np.mean(total_joint_losses)))

        if not continue_training:
            return
